# -*- coding: utf-8 -*-
"""
POMO Neural Network for Truck-Drone EVRP-TW.

Transformer encoder + attention decoder adapted from POMO CVRP.
Supports transfer learning from pre-trained CVRP checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Encoder ────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """Transformer encoder: MHA + Add&Norm + FF + Add&Norm."""

    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embedding_dim),
        )
        self.norm1 = nn.InstanceNorm2d(1, affine=True, track_running_stats=False)
        self.norm2 = nn.InstanceNorm2d(1, affine=True, track_running_stats=False)

    def forward(self, x):
        b, n, emb = x.shape
        h, qk = self.head_num, self.qkv_dim

        q = self.Wq(x).reshape(b, n, h, qk).permute(0, 2, 1, 3)
        k = self.Wk(x).reshape(b, n, h, qk).permute(0, 2, 1, 3)
        v = self.Wv(x).reshape(b, n, h, qk).permute(0, 2, 1, 3)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(qk)
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n, h * qk)
        out = self.multi_head_combine(out)
        x = self.norm1((x + out).unsqueeze(1)).squeeze(1)

        ff = self.feed_forward(x)
        x = self.norm2((x + ff).unsqueeze(1)).squeeze(1)
        return x


class EVRPEncoder(nn.Module):
    """Encodes depot + customer nodes with Transformer."""

    def __init__(self, embedding_dim=128, encoder_layer_num=6,
                 head_num=8, qkv_dim=16, ff_hidden=512,
                 node_feature_dim=6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_feature_dim = node_feature_dim

        self.depot_embed = nn.Linear(2, embedding_dim)
        self.node_embed = nn.Linear(node_feature_dim, embedding_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden)
            for _ in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_features):
        """depot_xy: (batch, 1, 2), node_features: (batch, N, node_feature_dim)"""
        d_emb = self.depot_embed(depot_xy)  # (batch, 1, emb)
        n_emb = self.node_embed(node_features)  # (batch, N, emb)
        x = torch.cat([d_emb, n_emb], dim=1)  # (batch, N+1, emb)
        for layer in self.layers:
            x = layer(x)
        return x


# ── Decoder ────────────────────────────────────────────────────────────

class EVRPDecoder(nn.Module):
    """Attention decoder: outputs probabilities over next node."""

    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16,
                 logit_clipping=10.0, context_dim=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.logit_clipping = logit_clipping

        # Query: last_node_emb + context (load, time, battery)
        q_input_dim = embedding_dim + context_dim
        self.Wq = nn.Linear(q_input_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Single-head for final logits
        self.single_head_Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.single_head_Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Cached K, V from encoded nodes
        self._k_mh = None
        self._v_mh = None
        self._single_k = None

    def set_kv(self, encoded_nodes):
        """Pre-compute keys and values from encoded nodes."""
        b, n, emb = encoded_nodes.shape
        h, qk = self.head_num, self.qkv_dim

        k_mh = self.Wk(encoded_nodes).reshape(b, n, h, qk).permute(0, 2, 1, 3)
        v_mh = self.Wv(encoded_nodes).reshape(b, n, h, qk).permute(0, 2, 1, 3)
        self._k_mh = k_mh  # (batch, head, N+1, qkv_dim)
        self._v_mh = v_mh
        self._single_k = self.single_head_Wk(encoded_nodes)  # (batch, N+1, emb)

    def forward(self, encoded_nodes, last_node_emb, context, ninf_mask):
        """
        last_node_emb: (batch, pomo, emb)
        context: (batch, pomo, context_dim)
        ninf_mask: (batch, pomo, N+1), 0=valid, -inf=masked
        """
        b_pomo, pomo, emb = last_node_emb.shape
        b_enc = encoded_nodes.shape[0]

        # Handle batch size mismatch (augmented data)
        if b_enc != b_pomo:
            encoded_nodes = encoded_nodes.expand(b_pomo, -1, -1)
            self._k_mh = self._k_mh.expand(b_pomo, -1, -1, -1)
            self._v_mh = self._v_mh.expand(b_pomo, -1, -1, -1)
            self._single_k = self._single_k.expand(b_pomo, -1, -1)

        q_input = torch.cat([last_node_emb, context], dim=-1)
        q_mh = self.Wq(q_input).reshape(b_pomo, pomo, self.head_num, self.qkv_dim)
        q_mh = q_mh.permute(0, 2, 1, 3)  # (batch, head, pomo, qkv_dim)

        score_mh = torch.matmul(q_mh, self._k_mh.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        score_mh = score_mh + ninf_mask.unsqueeze(1)
        attn_mh = F.softmax(score_mh, dim=-1)
        out_mh = torch.matmul(attn_mh, self._v_mh)
        out_mh = out_mh.permute(0, 2, 1, 3).reshape(b_pomo, pomo, self.head_num * self.qkv_dim)
        out_mh = self.multi_head_combine(out_mh)

        q_final = self.single_head_Wq(out_mh)
        logits = torch.matmul(q_final, self._single_k.transpose(1, 2))
        logits = logits / math.sqrt(self.embedding_dim)
        logits = self.logit_clipping * torch.tanh(logits)
        logits = logits + ninf_mask

        return F.softmax(logits, dim=-1)


# ── Full Model ─────────────────────────────────────────────────────────

class POMOModel(nn.Module):
    """Complete POMO model for EVRP-TW."""

    def __init__(self, embedding_dim=128, encoder_layer_num=6,
                 head_num=8, qkv_dim=16, ff_hidden=512,
                 logit_clipping=10.0, node_feature_dim=6, context_dim=3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = EVRPEncoder(
            embedding_dim=embedding_dim,
            encoder_layer_num=encoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden=ff_hidden,
            node_feature_dim=node_feature_dim,
        )
        self.decoder = EVRPDecoder(
            embedding_dim=embedding_dim,
            head_num=head_num,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
            context_dim=context_dim,
        )

    def pre_forward(self, depot_xy, node_features):
        """Encode problem once. depot_xy: (batch, 1, 2), node_features: (batch, N, F)"""
        self.encoded_nodes = self.encoder(depot_xy, node_features)
        self.decoder.set_kv(self.encoded_nodes)
        return self.encoded_nodes

    def forward(self, state):
        """
        One decoding step.
        state dict: last_node_idx, load, time, battery, ninf_mask
        """
        b, pomo = state['load'].shape
        enc = self.encoded_nodes
        emb = self.embedding_dim

        # Gather last node embedding
        last_idx = state['last_node_idx'].long().clamp(min=0)  # (batch, pomo)
        b_enc, n_nodes = enc.shape[0], enc.shape[1]

        enc_flat = enc.reshape(-1, emb)
        batch_offset = torch.arange(b_enc, device=enc.device).unsqueeze(1) * n_nodes
        gather_idx = (last_idx + batch_offset).reshape(-1)
        last_emb = enc_flat[gather_idx].reshape(b_enc, pomo, emb)

        # Build context: [load, time/horizon, battery]
        context = torch.stack([
            state['load'],
            state['time'] / 240.0,
            state['battery'],
        ], dim=-1)

        return self.decoder(enc, last_emb, context, state['ninf_mask'])

    # ── Transfer Learning ──────────────────────────────────────────

    def load_cvrp_pretrained(self, checkpoint_path, map_location='cpu'):
        """
        Transfer encoder weights from pre-trained POMO CVRP model.

        The CVRP model has:
          - embedding_depot: (128, 2) → matches exactly
          - embedding_node: (128, 3) → we have (128, 6), copy first 3 cols
          - 6 encoder layers: identical structure → copy all
          - Decoder: different architecture, skip (random init)
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        cvrp_state = ckpt['model_state_dict']

        # Map CVRP keys to our keys
        key_map = {
            'encoder.depot_embed.weight': 'embedding_depot.weight',
            'encoder.depot_embed.bias': 'embedding_depot.bias',
        }

        # Map encoder layers
        for i in range(6):
            for ours, cvrp in [
                (f'encoder.layers.{i}.Wq.weight', f'encoder.layers.{i}.Wq.weight'),
                (f'encoder.layers.{i}.Wk.weight', f'encoder.layers.{i}.Wk.weight'),
                (f'encoder.layers.{i}.Wv.weight', f'encoder.layers.{i}.Wv.weight'),
                (f'encoder.layers.{i}.multi_head_combine.weight', f'encoder.layers.{i}.multi_head_combine.weight'),
                (f'encoder.layers.{i}.multi_head_combine.bias', f'encoder.layers.{i}.multi_head_combine.bias'),
                (f'encoder.layers.{i}.norm1.weight', f'encoder.layers.{i}.add_n_normalization_1.norm.weight'),
                (f'encoder.layers.{i}.norm1.bias', f'encoder.layers.{i}.add_n_normalization_1.norm.bias'),
                (f'encoder.layers.{i}.feed_forward.0.weight', f'encoder.layers.{i}.feed_forward.W1.weight'),
                (f'encoder.layers.{i}.feed_forward.0.bias', f'encoder.layers.{i}.feed_forward.W1.bias'),
                (f'encoder.layers.{i}.feed_forward.2.weight', f'encoder.layers.{i}.feed_forward.W2.weight'),
                (f'encoder.layers.{i}.feed_forward.2.bias', f'encoder.layers.{i}.feed_forward.W2.bias'),
                (f'encoder.layers.{i}.norm2.weight', f'encoder.layers.{i}.add_n_normalization_2.norm.weight'),
                (f'encoder.layers.{i}.norm2.bias', f'encoder.layers.{i}.add_n_normalization_2.norm.bias'),
            ]:
                key_map[ours] = cvrp

        transferred = 0
        skipped = 0

        with torch.no_grad():
            for our_name, our_param in self.named_parameters():
                if our_name in key_map:
                    cvrp_name = key_map[our_name]
                    if cvrp_name in cvrp_state:
                        cvrp_weight = cvrp_state[cvrp_name]
                        if our_param.shape == cvrp_weight.shape:
                            our_param.copy_(cvrp_weight)
                            transferred += 1
                        else:
                            # Handle embedding_node: CVRP (128,3) → ours (128,6)
                            if 'node_embed.weight' in our_name:
                                our_param[:, :3].copy_(cvrp_weight)
                                transferred += 1
                            else:
                                skipped += 1
                    else:
                        skipped += 1

        print(f'  Transferred: {transferred} params, Skipped: {skipped} params')
        print(f'  (Encoder from CVRP, decoder randomly initialized)')


def get_encoding(encoded_nodes, node_index_to_pick):
    """Gather node embeddings by index."""
    b, n, emb = encoded_nodes.shape
    bp, pp = node_index_to_pick.shape
    flat = encoded_nodes.reshape(-1, emb)
    offset = torch.arange(bp, device=encoded_nodes.device).unsqueeze(1) * n
    idx = (node_index_to_pick + offset).reshape(-1)
    return flat[idx].reshape(bp, pp, emb)
