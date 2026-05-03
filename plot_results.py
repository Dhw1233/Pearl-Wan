#!/usr/bin/env python3
"""
PEARL-WAN Results Plotting Script
Reads experiment JSON results and generates visualization plots.
"""
import os
import sys
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='Plot PEARL-WAN experiment results')
    parser.add_argument('--exp_dir', type=str, default='exp', help='Experiment directory')
    parser.add_argument('--output_dir', type=str, default='exp/plots', help='Output directory for plots')
    return parser.parse_args()

def load_summary(exp_path, pattern="*_summary.json"):
    """Load all summary JSON files from an experiment directory."""
    summaries = {}
    files = glob.glob(os.path.join(exp_path, pattern))
    for f in files:
        basename = os.path.basename(f)
        with open(f) as fp:
            summaries[basename] = json.load(fp)
    # Also try loading eval_wan_results.json
    wan_results = os.path.join(exp_path, "eval_wan_results.json")
    if os.path.exists(wan_results):
        with open(wan_results) as fp:
            data = json.load(fp)
            for run in data.get("runs", []):
                mode = run.get("mode", "unknown")
                summaries[f"wan_{mode}"] = {
                    "eval_mode": mode,
                    "speed": run.get("avg_speed", 0),
                    "total_time": run.get("total_time", 0),
                    "total_tokens": run.get("total_tokens", 0),
                }
    return summaries

def plot_mode_comparison(exp_dirs, output_dir):
    """Plot speed comparison across different eval modes."""
    modes = []
    speeds = []
    speed_stds = []
    draft_forwards = []
    target_forwards = []
    
    for exp_name, exp_path in exp_dirs.items():
        summaries = load_summary(exp_path)
        for fname, data in summaries.items():
            mode = data.get("eval_mode", "unknown")
            if mode not in modes:
                modes.append(mode)
                speeds.append(data.get("speed", 0))
                speed_stds.append(data.get("speed_std", 0))
                draft_forwards.append(data.get("draft_forward_times", 0))
                target_forwards.append(data.get("target_forward_times", 0))
    
    if not modes:
        print("No summary data found for mode comparison.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Speed comparison
    x = np.arange(len(modes))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    axes[0].bar(x, speeds, yerr=speed_stds, color=colors[:len(modes)], capsize=5, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=15, ha='right')
    axes[0].set_ylabel('Speed (tokens / second)')
    axes[0].set_title('Inference Speed Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    for i, (v, s) in enumerate(zip(speeds, speed_stds)):
        axes[0].text(i, v + s + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Forward counts
    width = 0.35
    axes[1].bar(x - width/2, draft_forwards, width, label='Draft Model', color='#3498db', edgecolor='black')
    axes[1].bar(x + width/2, target_forwards, width, label='Target Model', color='#e74c3c', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, rotation=15, ha='right')
    axes[1].set_ylabel('Forward Count')
    axes[1].set_title('Model Forward Counts')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'mode_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_rtt_comparison(exp_dirs, output_dir):
    """Plot speed vs RTT for WAN experiments."""
    rtt_values = []
    speeds = []
    labels = []
    
    for exp_name, exp_path in exp_dirs.items():
        if 'rtt' not in exp_name.lower():
            continue
        summaries = load_summary(exp_path)
        for fname, data in summaries.items():
            # Extract RTT from exp name, e.g., pearl_wan_rtt_50
            parts = exp_name.split('_')
            for i, p in enumerate(parts):
                if p == 'rtt' and i + 1 < len(parts):
                    try:
                        rtt = float(parts[i+1])
                        rtt_values.append(rtt)
                        speeds.append(data.get("speed", 0))
                        labels.append(f"{data.get('eval_mode', 'wan')}")
                    except ValueError:
                        pass
    
    if not rtt_values:
        print("No RTT experiment data found.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = sorted(set(labels))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        x = [rtt_values[i] for i in range(len(labels)) if labels[i] == label]
        y = [speeds[i] for i in range(len(labels)) if labels[i] == label]
        if x:
            ax.plot(sorted(x), [y for _, y in sorted(zip(x, y))], 
                   marker='o', linewidth=2, markersize=8, 
                   label=label, color=colors[idx])
    
    ax.set_xlabel('RTT (ms)')
    ax.set_ylabel('Speed (tokens / second)')
    ax.set_title('WAN Inference Speed vs Network RTT')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'rtt_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_wan_stats(exp_path, output_dir):
    """Plot detailed PEARL-WAN statistics."""
    result_file = os.path.join(exp_path, "eval_wan_results.json")
    if not os.path.exists(result_file):
        return
    
    with open(result_file) as f:
        data = json.load(f)
    
    runs = data.get("runs", [])
    if not runs:
        return
    
    modes = [r["mode"] for r in runs]
    speeds = [r.get("avg_speed", 0) for r in runs]
    times = [r.get("total_time", 0) for r in runs]
    tokens = [r.get("total_tokens", 0) for r in runs]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Speed
    axes[0, 0].bar(modes, speeds, color=['#3498db', '#2ecc71', '#e74c3c'][:len(modes)], edgecolor='black')
    axes[0, 0].set_ylabel('Speed (tok/s)')
    axes[0, 0].set_title('Average Generation Speed')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(speeds):
        axes[0, 0].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    
    # Time
    axes[0, 1].bar(modes, times, color=['#3498db', '#2ecc71', '#e74c3c'][:len(modes)], edgecolor='black')
    axes[0, 1].set_ylabel('Total Time (s)')
    axes[0, 1].set_title('Total Generation Time')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Tokens
    axes[1, 0].bar(modes, tokens, color=['#3498db', '#2ecc71', '#e74c3c'][:len(modes)], edgecolor='black')
    axes[1, 0].set_ylabel('Total Tokens')
    axes[1, 0].set_title('Total Generated Tokens')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Speedup ratio
    if len(speeds) >= 2 and speeds[0] > 0:
        speedups = [s / speeds[0] for s in speeds]
        axes[1, 1].bar(modes, speedups, color=['#3498db', '#2ecc71', '#e74c3c'][:len(modes)], edgecolor='black')
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline')
        axes[1, 1].set_ylabel('Speedup Ratio')
        axes[1, 1].set_title('Speedup vs Autoregressive')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(speedups):
            axes[1, 1].text(i, v + 0.02, f'{v:.2f}x', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    exp_name = os.path.basename(exp_path)
    out_path = os.path.join(output_dir, f'{exp_name}_wan_stats.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_category_breakdown(exp_dirs, output_dir):
    """Plot MGSM category breakdown if available."""
    for exp_name, exp_path in exp_dirs.items():
        pattern = os.path.join(exp_path, "*_mgsm_summary.json")
        files = glob.glob(pattern)
        if not files:
            continue
    
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        
        category_speeds = data.get("category_speeds", {})
        if not category_speeds:
            continue
        
        categories = list(category_speeds.keys())
        speeds = list(category_speeds.values())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        bars = ax.bar(categories, speeds, color=colors, edgecolor='black')
        ax.set_ylabel('Speed (tokens / second)')
        ax.set_title(f'MGSM Category Speed Breakdown ({data.get("eval_mode", "unknown")})')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=30, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        basename = os.path.basename(f).replace('.json', '_categories.png')
        out_path = os.path.join(output_dir, basename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover all experiment directories
    exp_dirs = {}
    for item in os.listdir(args.exp_dir):
        item_path = os.path.join(args.exp_dir, item)
        if os.path.isdir(item_path):
            exp_dirs[item] = item_path
    
    print(f"Found {len(exp_dirs)} experiment directories: {list(exp_dirs.keys())}")
    
    # Generate plots
    plot_mode_comparison(exp_dirs, args.output_dir)
    plot_rtt_comparison(exp_dirs, args.output_dir)
    plot_category_breakdown(exp_dirs, args.output_dir)
    
    # Plot WAN stats for each experiment
    for exp_name, exp_path in exp_dirs.items():
        plot_wan_stats(exp_path, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
