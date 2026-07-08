# -*- coding: utf-8 -*-
"""
POMO Trainer for EVRP-TW — optimized for MPS, with transfer learning support.

REINFORCE with POMO shared baseline.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from algorithms.pomo.pomo_model import POMOModel
from algorithms.pomo.pomo_env import POMOEnv
from algorithms.pomo.pomo_problem import generate_random_problems


class POMOTrainer:
    """Trains POMO model with REINFORCE + POMO baseline."""

    def __init__(self, model, problem_size=50, batch_size=64,
                 lr=1e-4, weight_decay=1e-6, device='cpu',
                 checkpoint_dir=None,
                 truck_speed=35.0, battery_capacity=100.0,
                 energy_per_km=1.0, tw_horizon=240.0):
        self.model = model.to(device)
        self.device = device
        self.problem_size = problem_size
        self.batch_size = batch_size

        self.checkpoint_dir = checkpoint_dir or './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.env = POMOEnv(
            truck_speed=truck_speed, battery_capacity=battery_capacity,
            energy_per_km=energy_per_km, tw_horizon=tw_horizon,
        )

        self.epoch = 0
        self.best_score = float('inf')

    def train_one_batch(self, problems):
        """
        One training batch with POMO REINFORCE.

        POMO init:
          Step 0: All to depot
          Step 1: Trajectory i → customer i+1 (different start per trajectory)
          Steps 2+: Model sampling
        """
        self.model.train()
        device = self.device

        self.env.load_problems(problems)
        state = self.env.reset(device=device)

        # Build node features
        node_feat = torch.zeros(len(problems), self.problem_size, 6, device=device)
        depot_xy = torch.zeros(len(problems), 1, 2, device=device)
        for i, p in enumerate(problems):
            node_feat[i, :, :2] = p['node_xy'].to(device)
            node_feat[i, :, 2] = p['node_demand'].to(device)
            node_feat[i, :, 3] = p['node_tw_start'].to(device)
            node_feat[i, :, 4] = p['node_tw_end'].to(device)
            node_feat[i, :, 5] = p['node_service'].to(device)
            d = p['depot_xy'].to(device)
            depot_xy[i, 0, :] = d.reshape(-1)[:2]

        self.model.pre_forward(depot_xy, node_feat)

        b, pomo = self.env.batch_size, self.env.pomo_size

        # Step 0: All to depot
        self.env.step(torch.zeros((b, pomo), dtype=torch.long, device=device))
        state = self.env._get_state()

        # Step 1: POMO init — each trajectory to different customer
        step1 = torch.arange(1, pomo + 1, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        self.env.step(step1)
        state = self.env._get_state()

        # Steps 2+: Model rollouts
        prob_list = []
        done = torch.zeros((b, pomo), dtype=torch.bool, device=device)
        max_steps = pomo * 3

        for _ in range(max_steps):
            if done.all():
                break
            probs = self.model(state)
            # Sample actions
            flat_probs = probs.reshape(-1, probs.shape[-1])
            selected = flat_probs.multinomial(1).reshape(b, pomo)
            prob_sel = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
            prob_list.append(prob_sel)
            step_done = self.env.step(selected)
            done = done | step_done
            state = self.env._get_state()

        # Reward
        reward = self.env.get_reward()  # (batch, pomo), already negative cost

        # POMO baseline
        advantage = reward - reward.mean(dim=1, keepdim=True)

        # Log-prob
        log_prob = torch.stack(prob_list, dim=2).sum(dim=2) if prob_list else torch.zeros_like(reward)
        log_prob = torch.where(torch.isinf(log_prob) | torch.isnan(log_prob),
                              torch.zeros_like(log_prob), log_prob)

        # REINFORCE loss
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        self.optimizer.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            best = reward.max(dim=1).values
            score = -best.mean().item()

        return score, loss_mean.item()

    def train_epoch(self, num_episodes=200, tw_type='mixed', log_interval=50):
        """Train one epoch."""
        scores, losses = [], []
        t0 = time.time()

        for ep in range(num_episodes):
            problems = generate_random_problems(self.batch_size, self.problem_size, tw_type=tw_type)
            try:
                s, l = self.train_one_batch(problems)
                scores.append(s)
                losses.append(l)
            except RuntimeError as e:
                print(f"  Ep {ep} error: {e}")
                continue

            if (ep + 1) % log_interval == 0:
                avg_s = np.mean(scores[-log_interval:]) if scores else 0
                avg_l = np.mean(losses[-log_interval:]) if losses else 0
                elapsed = time.time() - t0
                print(f"    Ep {ep+1:4d}/{num_episodes} | Cost: {avg_s:.1f} | "
                      f"Loss: {avg_l:.4f} | {elapsed:.1f}s")

        return (np.mean(scores) if scores else float('inf'),
                np.mean(losses) if losses else 0)

    def save_checkpoint(self, name=None):
        if name is None:
            name = f'checkpoint_epoch{self.epoch:04d}.pt'
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
        }, path)
        print(f"  Saved: {os.path.basename(path)}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt.get('epoch', 0)
        self.best_score = ckpt.get('best_score', float('inf'))
        print(f"  Loaded checkpoint epoch {self.epoch}")

    def run(self, epochs=50, episodes_per_epoch=200, tw_type='mixed',
            lr_milestones=None, gamma=0.1, save_interval=10):
        """Full training loop."""
        if lr_milestones is None:
            lr_milestones = [int(epochs * 0.75), int(epochs * 0.95)]

        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=gamma)

        print(f"POMO Training | Device: {self.device} | Size: {self.problem_size}")
        print(f"  Batch: {self.batch_size} | LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"  Epochs: {epochs} | Ep/epoch: {episodes_per_epoch}")
        print(f"  Total batches: {epochs * episodes_per_epoch}")

        history = {'epoch': [], 'cost': [], 'loss': [], 'lr': []}

        for ep in range(self.epoch, epochs):
            self.epoch = ep
            t0 = time.time()
            print(f"\nEpoch {ep+1}/{epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")

            avg_cost, avg_loss = self.train_epoch(episodes_per_epoch, tw_type, log_interval=max(episodes_per_epoch//4, 1))
            scheduler.step()

            elapsed = time.time() - t0
            print(f"  Done | Cost: {avg_cost:.1f} | Loss: {avg_loss:.4f} | {elapsed:.0f}s")

            history['epoch'].append(ep)
            history['cost'].append(float(avg_cost))
            history['loss'].append(float(avg_loss))
            history['lr'].append(float(scheduler.get_last_lr()[0]))

            if avg_cost < self.best_score:
                self.best_score = avg_cost
                self.save_checkpoint('best_model.pt')
            if (ep + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch{ep+1:04d}.pt')

        self.save_checkpoint('final_model.pt')
        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

        print(f"\nTraining done. Best cost: {self.best_score:.1f}")
        return history
