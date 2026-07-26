#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POMO Training Entry Point -- MPS-optimized with transfer learning.

Usage:
    # Train with transfer learning (recommended, ~1-3 hours on M4)
    python -m src.algorithms.pomo.train --pretrained --epochs 50

    # Train from scratch (slower, ~6-12 hours)
    python -m src.algorithms.pomo.train --epochs 100

    # CPU training
    python -m src.algorithms.pomo.train --device cpu --epochs 30

    # Quick test
    python -m src.algorithms.pomo.train --epochs 5 --episodes-per-epoch 50
"""

import os
import sys
import argparse

import torch
import multiprocessing

from src.algorithms.pomo.pomo_model import POMOModel
from src.algorithms.pomo.pomo_trainer import POMOTrainer
from src.config import TRUCK_SPEED, BATTERY_CAPACITY


def get_device(device_str):
    """Resolve device."""
    if device_str == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        print("MPS not available, using CPU")
    elif device_str == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("CUDA not available, using CPU")
    return torch.device('cpu')


def find_cvrp_checkpoint():
    """Find pre-trained CVRP model in POMO folder."""
    _pomo_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(_pomo_dir, '..', 'POMO', 'NEW_py_ver', 'CVRP', 'POMO',
                     'result', 'saved_CVRP100_model', 'checkpoint-30500.pt'),
        os.path.join(_pomo_dir, '..', 'POMO', 'OLD_ipynb_ver', 'POMO_CVRP',
                     'result', 'Saved_CVRP100_Model', 'ACTOR_state_dic.pt'),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='POMO Training for EVRP-TW')

    # Model
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--encoder-layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)

    # Training
    parser.add_argument('--epochs', type=int, default=80,
                       help='Training epochs (~6-8 hrs with pretrained on CPU)')
    parser.add_argument('--episodes-per-epoch', type=int, default=400,
                       help='Batches per epoch')
    parser.add_argument('--problem-size', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tw-type', type=str, default='mixed',
                       choices=['RC1', 'RC2', 'mixed'])

    # Transfer learning (ON by default!)
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Transfer learning from CVRP (default: True)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch')
    parser.add_argument('--pretrained-path', type=str, default=None)

    # Device (CPU is faster for this model size)
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'])

    # Checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=10)

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # CPU threading
    if device.type == 'cpu':
        n_threads = min(multiprocessing.cpu_count(), 8)
        torch.set_num_threads(n_threads)
        print(f"CPU threads: {n_threads}")

    # Checkpoint dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    # Create model
    model = POMOModel(
        embedding_dim=args.embedding_dim,
        encoder_layer_num=args.encoder_layers,
        head_num=args.heads,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    if args.no_pretrained:
        args.pretrained = False

    # Transfer learning
    if args.pretrained:
        cvrp_path = args.pretrained_path or find_cvrp_checkpoint()
        if cvrp_path and os.path.exists(cvrp_path):
            print(f"Loading CVRP weights: {cvrp_path}")
            model.load_cvrp_pretrained(cvrp_path, map_location='cpu')
            print(f"  (Fine-tuning for {args.epochs} epochs)")
        else:
            print("WARNING: No CVRP checkpoint found, training from scratch")
            print(f"  Consider --epochs 100+ for scratch training")

    # Trainer
    trainer = POMOTrainer(
        model=model, problem_size=args.problem_size,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=1e-6, device=device,
        checkpoint_dir=args.checkpoint_dir,
        truck_speed=TRUCK_SPEED, battery_capacity=BATTERY_CAPACITY,
        energy_per_km=1.0, tw_horizon=240.0,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.run(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        tw_type=args.tw_type,
        lr_milestones=[int(args.epochs * 0.75), int(args.epochs * 0.95)],
        gamma=0.1,
        save_interval=args.save_interval,
    )

    print(f"\nModel saved to: {args.checkpoint_dir}/best_model.pt")
    print(f"Run experiments: python run_pomo_experiments.py")


if __name__ == '__main__':
    main()
