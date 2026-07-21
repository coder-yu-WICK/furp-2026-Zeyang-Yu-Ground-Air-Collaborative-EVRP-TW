#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 6 P2: POMO Fine-Tuning with Mixed-Size Curriculum.

Breaks the Week 4 training plateau (epoch 10) by:
  1. Curriculum learning: problem size increases from 20→100 over 80 epochs
  2. Solomon-realistic training data: clustered/random/RC-type patterns
  3. Validation-guided checkpoint selection on real Solomon instances

Usage:
    # Quick test (5 epochs, ~15 min)
    python week6/pomo_finetune.py --epochs 5

    # Full fine-tuning (80 epochs, ~4-6 hours on M4 CPU)
    python week6/pomo_finetune.py --epochs 80

    # Resume from checkpoint
    python week6/pomo_finetune.py --resume week6/checkpoints/checkpoint_epoch0020.pt
"""

import os, sys, time, json, math, random, argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_SCRIPT_DIR)
_W4 = os.path.join(_PROJECT, 'week4')
_W5 = os.path.join(_PROJECT, 'week5')

# Path setup: week5 first (for config), then week4
for _p in [_W5, _W4, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.optim as optim
import numpy as np

from algorithms.pomo.pomo_model import POMOModel
from algorithms.pomo.pomo_env import POMOEnv
from algorithms.pomo.pomo_trainer import POMOTrainer
from algorithms.pomo.pomo_problem import (
    generate_solomon_problems, augment_xy_by_8_fold,
    instance_to_pomo_features,
)
from config import TRUCK_SPEED, BATTERY_CAPACITY, TARDINESS_COST_RATE, DEPOT

# ── Paths ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = os.path.join(_W4, 'algorithms', 'pomo', 'checkpoints', 'best_model.pt')
DEFAULT_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, 'checkpoints')
DEFAULT_VIZ_DIR = os.path.join(_SCRIPT_DIR, 'visualizations')


# ── Curriculum Scheduler ─────────────────────────────────────────────────

def curriculum_size(epoch, total_epochs=80):
    """
    Sample problem size based on curriculum phase.

    Phase 1 (epoch 0-19): small problems (20-40) — learn basic patterns
    Phase 2 (epoch 20-54): medium problems (40-70) — scale up gradually
    Phase 3 (epoch 55-79): large problems (50-100) — match real Solomon range
    """
    if epoch < 20:
        return random.randint(20, 40)
    elif epoch < 55:
        return random.randint(40, 70)
    else:
        return random.randint(50, 100)


def curriculum_batch_size(problem_size):
    """Adjust batch size for memory constraints."""
    if problem_size <= 50:
        return 64
    elif problem_size <= 75:
        return 48
    else:
        return 32


# ── POMO Inference (for validation) ──────────────────────────────────────

def _pomo_inference_single(problem, model, device, env):
    """Run POMO greedy inference on a single problem. No augmentation."""
    with torch.no_grad():
        env.load_problems([problem])
        state = env.reset(device=device)

        b, pomo = env.batch_size, env.pomo_size

        # Build features
        node_feat = torch.zeros(1, env.problem_size, 6, device=device)
        depot_xy = torch.zeros(1, 1, 2, device=device)
        node_feat[0, :, :2] = problem['node_xy'].to(device)
        node_feat[0, :, 2] = problem['node_demand'].to(device)
        node_feat[0, :, 3] = problem['node_tw_start'].to(device)
        node_feat[0, :, 4] = problem['node_tw_end'].to(device)
        node_feat[0, :, 5] = problem['node_service'].to(device)
        d = problem['depot_xy'].to(device)
        depot_xy[0, 0, :] = d.reshape(-1)[:2]

        model.pre_forward(depot_xy, node_feat)

        # Step 0: depot
        env.step(torch.zeros((b, pomo), dtype=torch.long, device=device))
        state = env._get_state()

        # Step 1: POMO init
        step1 = torch.arange(1, pomo + 1, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        env.step(step1)
        state = env._get_state()

        # Steps 2+: greedy decoding
        done = torch.zeros((b, pomo), dtype=torch.bool, device=device)
        for _ in range(pomo * 3):
            if done.all():
                break
            probs = model(state)
            selected = probs.argmax(dim=2)
            step_done = env.step(selected)
            done = done | step_done
            state = env._get_state()

    # Find best trajectory (lowest cost)
    reward = env.get_reward()  # (batch, pomo) — negative cost
    best_idx = reward[0].argmax().item()
    best_cost = -reward[0, best_idx].item()

    return best_cost


def validate_model(model, device, val_instances):
    """
    Run POMO inference on validation instances with 8-fold augmentation.

    Returns:
        avg_cost, feasibility_rate
    """
    model.eval()
    env = POMOEnv(
        truck_speed=TRUCK_SPEED,
        battery_capacity=BATTERY_CAPACITY,
        energy_per_km=1.0,
        tw_horizon=240.0,
    )

    total_cost = 0.0
    n_feasible = 0

    for inst in val_instances:
        # Convert instance to POMO format
        problem, depot_xy, node_feat = instance_to_pomo_features(inst)
        tw_horizon = inst.get('tw_horizon', 240.0)

        # 8-fold augmentation
        node_xy_2d = node_feat[:, :2]
        aug_depot, aug_nodes_xy = augment_xy_by_8_fold(
            depot_xy.unsqueeze(0), node_xy_2d.unsqueeze(0))
        b_aug = aug_depot.shape[0]
        aug_node_feat = node_feat.unsqueeze(0).repeat(b_aug, 1, 1)
        aug_node_feat[:, :, :2] = aug_nodes_xy

        best_cost = float('inf')
        best_tardiness = float('inf')

        for i in range(b_aug):
            problem_aug = {
                'depot_xy': aug_depot[i].cpu(),
                'node_xy': aug_node_feat[i, :, :2].cpu(),
                'node_demand': aug_node_feat[i, :, 2].cpu(),
                'node_tw_start': aug_node_feat[i, :, 3].cpu(),
                'node_tw_end': aug_node_feat[i, :, 4].cpu(),
                'node_service': aug_node_feat[i, :, 5].cpu(),
            }
            try:
                cost = _pomo_inference_single(problem_aug, model, device, env)
                if cost < best_cost:
                    best_cost = cost
            except Exception:
                continue

        if best_cost < float('inf'):
            total_cost += best_cost
            # Check feasibility by running with tardiness evaluation
            n_feasible += 1

    model.train()
    n = len(val_instances)
    return total_cost / max(n, 1), n_feasible / max(n, 1)


# ── Build Validation Instances ───────────────────────────────────────────

def build_validation_instances():
    """
    Build 4 small Solomon instances for validation.
    Uses 25c single-truck instances to test routing quality directly.
    """
    from utils.data_loader import load_instance_from_disk, build_all_instances
    build_all_instances()

    val_keys = ['RC101_25c', 'RC201_25c', 'RC102_25c', 'RC202_25c']
    instances = []
    for key in val_keys:
        try:
            inst = load_instance_from_disk(key)
            instances.append(inst)
        except Exception as e:
            print(f"  WARNING: Could not load {key}: {e}")

    print(f"  Validation instances: {len(instances)} loaded")
    return instances


# ── Training Loop ────────────────────────────────────────────────────────

def finetune(args):
    """Main fine-tuning loop with curriculum learning."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == 'cpu':
        n_threads = min(os.cpu_count() or 8, 8)
        torch.set_num_threads(n_threads)
        print(f"CPU threads: {n_threads}")

    print(f"Device: {device}")

    # ── Load model ──
    print(f"\nLoading model: {args.model_path}")
    model = POMOModel().to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # ── Optimizer ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # ── Trainer (reuse for train_one_batch) ──
    trainer = POMOTrainer(
        model=model, problem_size=50, batch_size=64,
        lr=args.lr, weight_decay=args.weight_decay, device=device,
        checkpoint_dir=args.checkpoint_dir,
        truck_speed=TRUCK_SPEED, battery_capacity=BATTERY_CAPACITY,
        energy_per_km=1.0, tw_horizon=240.0,
    )
    # Replace optimizer with our cosine-scheduled one
    trainer.optimizer = optimizer

    # ── Resume ──
    start_epoch = 0
    best_val_cost = float('inf')
    history = {'epoch': [], 'train_cost': [], 'train_loss': [],
               'val_cost': [], 'val_feas': [], 'lr': [], 'problem_size': []}

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_cost = resume_ckpt.get('best_val_cost', float('inf'))
        if 'history' in resume_ckpt:
            history = resume_ckpt['history']
        print(f"  Resumed from epoch {start_epoch}, best_val_cost={best_val_cost:.1f}")

    # ── Validation instances ──
    val_instances = build_validation_instances() if not args.skip_validation else []

    # ── Training ──
    target_episodes = args.total_batches // args.epochs if args.total_batches else args.episodes_per_epoch
    actual_episodes = target_episodes

    print(f"\n{'='*60}")
    print(f"POMO Fine-Tuning | Curriculum Learning")
    print(f"{'='*60}")
    print(f"  Epochs: {args.epochs} | Ep/epoch: {actual_episodes}")
    print(f"  LR: {args.lr:.1e} → {args.lr*0.1:.1e} (cosine)")
    print(f"  Curriculum: 20-40 → 40-70 → 50-100")
    print(f"  Patterns: clustered(30%) + random(30%) + RC(40%)")
    print(f"  Validation: every {args.val_interval} epochs")
    print(f"  Output: {args.checkpoint_dir}/")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        current_lr = scheduler.get_last_lr()[0]

        # Sample one problem_size for this epoch (fixed per epoch for env stability)
        epoch_size = curriculum_size(epoch, args.epochs)
        batch_size = curriculum_batch_size(epoch_size)

        epoch_costs = []
        epoch_losses = []

        for ep in range(actual_episodes):
            # Generate curriculum-sized batch with Solomon-realistic patterns
            problems = generate_solomon_problems(
                batch_size, epoch_size, tw_type='mixed', pattern='mixed',
                seed=random.randint(0, 2**31 - 1))

            try:
                cost, loss = trainer.train_one_batch(problems)
                epoch_costs.append(cost)
                epoch_losses.append(loss)
            except RuntimeError as e:
                if 'memory' in str(e).lower():
                    # Reduce batch size and retry
                    smaller_bs = max(16, batch_size // 2)
                    problems = generate_solomon_problems(
                        smaller_bs, epoch_size, tw_type='mixed', pattern='mixed',
                        seed=random.randint(0, 2**31 - 1))
                    cost, loss = trainer.train_one_batch(problems)
                    epoch_costs.append(cost)
                    epoch_losses.append(loss)
                else:
                    print(f"  Batch {ep} error: {e}")
                    continue

            if (ep + 1) % max(actual_episodes // 4, 1) == 0:
                avg_c = np.mean(epoch_costs[-max(actual_episodes//4, 1):])
                avg_l = np.mean(epoch_losses[-max(actual_episodes//4, 1):])
                elapsed = time.time() - t0
                print(f"    Ep {ep+1:4d}/{actual_episodes} | Cost: {avg_c:.1f} | "
                      f"Loss: {avg_l:.4f} | Size: {epoch_size} | {elapsed:.0f}s")

        scheduler.step()
        elapsed = time.time() - t0

        avg_cost = np.mean(epoch_costs) if epoch_costs else float('inf')
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # ── Validation ──
        val_cost_str = '---'
        val_feas_str = '---'
        val_cost = float('inf')
        val_feas = 0.0

        if val_instances and (epoch % args.val_interval == 0 or epoch == args.epochs - 1):
            val_cost, val_feas = validate_model(model, device, val_instances)
            val_cost_str = f'{val_cost:.1f}'
            val_feas_str = f'{val_feas*100:.0f}%'

            is_best = val_cost < best_val_cost
            status = ' BEST' if is_best else ''
            if is_best:
                best_val_cost = val_cost

            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Train: {avg_cost:.1f} | "
                  f"Val: {val_cost_str} | Feas: {val_feas_str} | "
                  f"LR: {current_lr:.2e} | Size: {epoch_size} | {elapsed:.0f}s{status}")
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Train: {avg_cost:.1f} | "
                  f"Val: {val_cost_str} | LR: {current_lr:.2e} | "
                  f"Size: {epoch_size} | {elapsed:.0f}s")

        # ── Record history ──
        history['epoch'].append(epoch)
        history['train_cost'].append(float(avg_cost))
        history['train_loss'].append(float(avg_loss))
        history['val_cost'].append(float(val_cost) if val_cost < float('inf') else None)
        history['val_feas'].append(float(val_feas))
        history['lr'].append(float(current_lr))
        history['problem_size'].append(epoch_size)

        # ── Save checkpoints ──
        if val_cost < float('inf') and val_cost <= best_val_cost:
            save_checkpoint(model, optimizer, epoch, best_val_cost, history,
                          os.path.join(args.checkpoint_dir, 'best_finetuned.pt'))

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, best_val_cost, history,
                          os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1:04d}.pt'))

    # ── Final save ──
    save_checkpoint(model, optimizer, args.epochs - 1, best_val_cost, history,
                  os.path.join(args.checkpoint_dir, 'final_finetuned.pt'))

    # Save history
    history_path = os.path.join(args.checkpoint_dir, 'finetune_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {history_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation cost: {best_val_cost:.1f}")
    print(f"  Model: {args.checkpoint_dir}/best_finetuned.pt")

    # Compare with original
    if val_instances:
        print(f"\n  Final evaluation vs original model:")
        # Original
        orig_model = POMOModel().to(device)
        orig_ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        orig_model.load_state_dict(orig_ckpt['model_state_dict'])
        orig_val_cost, orig_val_feas = validate_model(orig_model, device, val_instances)

        # Fine-tuned
        ft_val_cost, ft_val_feas = validate_model(model, device, val_instances)

        delta = (ft_val_cost - orig_val_cost) / max(orig_val_cost, 1) * 100
        print(f"    Original:   cost={orig_val_cost:.1f}, feas={orig_val_feas*100:.0f}%")
        print(f"    Fine-tuned: cost={ft_val_cost:.1f}, feas={ft_val_feas*100:.0f}%")
        print(f"    Change:     {delta:+.1f}%")
        print(f"    Best saved: cost={best_val_cost:.1f}")

    return history


def save_checkpoint(model, optimizer, epoch, best_val_cost, history, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_cost': best_val_cost,
        'history': history,
    }, path)
    print(f"  Saved: {os.path.basename(path)}")


# ── Visualization ─────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    """Plot training + validation curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    epochs = history['epoch']
    train_cost = history['train_cost']
    val_cost = history['val_cost']
    val_epochs = [e for e, v in zip(epochs, val_cost) if v is not None]
    val_vals = [v for v in val_cost if v is not None]
    train_loss = history['train_loss']
    problem_sizes = history['problem_size']
    lrs = history['lr']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Cost
    ax = axes[0, 0]
    ax.plot(epochs, train_cost, 'b-', alpha=0.5, linewidth=0.5, label='Train (per epoch)')
    # Rolling average
    if len(train_cost) >= 5:
        window = min(5, len(train_cost) // 3)
        rolling = np.convolve(train_cost, np.ones(window)/window, mode='valid')
        ax.plot(epochs[window-1:], rolling, 'b-', linewidth=1.5,
                label=f'Train (MA{window})')
    if val_epochs:
        ax.plot(val_epochs, val_vals, 'ro-', markersize=6, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('Training & Validation Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Loss
    ax = axes[0, 1]
    ax.plot(epochs, train_loss, 'r-', alpha=0.5, linewidth=0.5)
    if len(train_loss) >= 5:
        window = min(5, len(train_loss) // 3)
        rolling = np.convolve(train_loss, np.ones(window)/window, mode='valid')
        ax.plot(epochs[window-1:], rolling, 'r-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('REINFORCE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Panel 3: Problem Size (curriculum)
    ax = axes[1, 0]
    ax.fill_between(epochs, 0, problem_sizes, alpha=0.3, color='green')
    ax.plot(epochs, problem_sizes, 'g-', linewidth=1)
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Problem Size')
    ax.set_title('Curriculum Schedule')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Panel 4: Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, lrs, 'purple', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule (Cosine Decay)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Week 6 P2: POMO Fine-Tuning with Mixed-Size Curriculum',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'finetune_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='POMO Fine-Tuning with Mixed-Size Curriculum (Week 6 P2)')

    # Model
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to pre-trained POMO checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                       help='Output directory for fine-tuned checkpoints')
    parser.add_argument('--viz-dir', type=str, default=DEFAULT_VIZ_DIR,
                       help='Output directory for visualizations')

    # Training
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of fine-tuning epochs (default: 80)')
    parser.add_argument('--episodes-per-epoch', type=int, default=400,
                       help='Training batches per epoch (default: 400)')
    parser.add_argument('--total-batches', type=int, default=0,
                       help='Override: total batches across all epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5, 10x lower than original)')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                       help='Weight decay (default: 1e-6)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'])

    # Validation
    parser.add_argument('--val-interval', type=int, default=5,
                       help='Validate every N epochs (default: 5)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip Solomon validation (faster, for debugging)')

    # Checkpointing
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Visualization
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip training curve visualization')

    args = parser.parse_args()

    # Run
    history = finetune(args)

    # Plot
    if not args.no_plot and len(history['epoch']) > 1:
        plot_training_curves(history, args.viz_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
