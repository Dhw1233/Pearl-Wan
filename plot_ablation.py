#!/usr/bin/env python3
"""
PEARL-WAN Ablation Study Visualization
Reads experiment JSON results and generates ablation-specific plots.
"""
import os
import sys
import json
import glob
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='Plot PEARL-WAN ablation results')
    parser.add_argument('--exp_dir', type=str, default='exp', help='Experiment directory')
    parser.add_argument('--output_dir', type=str, default='exp/ablation_plots', help='Output directory for plots')
    return parser.parse_args()


def load_eval_wan_results(exp_path):
    """Load eval_wan_results.json from an experiment directory."""
    result_file = os.path.join(exp_path, "eval_wan_results.json")
    if not os.path.exists(result_file):
        return None
    with open(result_file) as f:
        return json.load(f)


def parse_exp_name(exp_name):
    """Parse ablation experiment name to extract metadata."""
    # Format: ablation_<draft>_<target>_rtt<rtt>_<abl>_<timestamp>
    # e.g., ablation_qwen2.5-0.5b-instruct_qwen2.5-1.5b-instruct_rtt20_full_1234567890
    pattern = r'ablation_(.+?)_(.+?)_rtt(\d+)_(full|no_adaptive|no_fallback|no_compression)_(\d+)'
    m = re.match(pattern, exp_name)
    if not m:
        return None
    draft, target, rtt, ablation, ts = m.groups()
    # Determine model pair label
    if '7b' in target.lower():
        model_label = '1.5B → 7B'
    else:
        model_label = '0.5B → 1.5B'
    return {
        'draft': draft,
        'target': target,
        'rtt': int(rtt),
        'ablation': ablation,
        'model_label': model_label,
        'timestamp': ts,
    }


def extract_speeds(data):
    """Extract avg_speed for autoregressive, speculative_decoding, wan from eval_wan_results."""
    speeds = {}
    for run in data.get('runs', []):
        mode = run.get('mode')
        speeds[mode] = run.get('avg_speed', 0)
    return speeds


def discover_ablation_experiments(exp_dir):
    """Discover all ablation experiment directories."""
    experiments = []
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if not os.path.isdir(item_path):
            continue
        meta = parse_exp_name(item)
        if not meta:
            continue
        data = load_eval_wan_results(item_path)
        if not data:
            continue
        meta['speeds'] = extract_speeds(data)
        meta['exp_name'] = item
        experiments.append(meta)
    return experiments


def plot_rtt_sweep(experiments, output_dir):
    """Plot WAN speed vs RTT for each (model, ablation) combination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ablation_colors = {
        'full': '#2ecc71',
        'no_adaptive': '#e74c3c',
        'no_fallback': '#3498db',
        'no_compression': '#9b59b6',
    }
    ablation_labels = {
        'full': 'Full',
        'no_adaptive': 'No Adaptive',
        'no_fallback': 'No Fallback',
        'no_compression': 'No Compression',
    }

    for idx, model_label in enumerate(['0.5B → 1.5B', '1.5B → 7B']):
        ax = axes[idx]
        model_exps = [e for e in experiments if e['model_label'] == model_label]
        for abl in ['full', 'no_adaptive', 'no_fallback', 'no_compression']:
            points = [(e['rtt'], e['speeds'].get('wan', 0)) for e in model_exps if e['ablation'] == abl]
            if not points:
                continue
            points = sorted(points, key=lambda x: x[0])
            x, y = zip(*points)
            ax.plot(x, y, marker='o', linewidth=2, markersize=8,
                   label=ablation_labels[abl], color=ablation_colors[abl])
        ax.set_xlabel('RTT (ms)')
        ax.set_ylabel('WAN Speed (tok/s)')
        ax.set_title(f'WAN Speed vs RTT ({model_label})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'rtt_sweep.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_ablation_comparison(experiments, output_dir):
    """Plot speed comparison across ablations for each (model, RTT)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ablations = ['full', 'no_adaptive', 'no_fallback', 'no_compression']
    ablation_labels = {
        'full': 'Full',
        'no_adaptive': 'No Adaptive',
        'no_fallback': 'No Fallback',
        'no_compression': 'No Compression',
    }
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    rtts = sorted(set(e['rtt'] for e in experiments))
    models = ['0.5B → 1.5B', '1.5B → 7B']

    for i, model in enumerate(models):
        for j, rtt in enumerate(rtts):
            ax = axes[i, j]
            model_rtt_exps = [e for e in experiments if e['model_label'] == model and e['rtt'] == rtt]
            speeds = []
            labels = []
            bar_colors = []
            for abl in ablations:
                exps = [e for e in model_rtt_exps if e['ablation'] == abl]
                if exps:
                    speed = exps[0]['speeds'].get('wan', 0)
                else:
                    speed = 0
                speeds.append(speed)
                labels.append(ablation_labels[abl])
                bar_colors.append(colors[ablations.index(abl)])

            x = np.arange(len(labels))
            bars = ax.bar(x, speeds, color=bar_colors, edgecolor='black', width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel('WAN Speed (tok/s)')
            ax.set_title(f'{model} @ RTT={rtt}ms')
            ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_model_comparison(experiments, output_dir):
    """Plot model pair comparison for each (RTT, ablation)."""
    rtts = sorted(set(e['rtt'] for e in experiments))
    ablations = ['full', 'no_adaptive', 'no_fallback', 'no_compression']
    ablation_labels = {
        'full': 'Full',
        'no_adaptive': 'No Adaptive',
        'no_fallback': 'No Fallback',
        'no_compression': 'No Compression',
    }

    fig, axes = plt.subplots(len(rtts), len(ablations), figsize=(16, 10))
    if len(rtts) == 1:
        axes = np.array([axes])

    for i, rtt in enumerate(rtts):
        for j, abl in enumerate(ablations):
            ax = axes[i, j]
            exps = [e for e in experiments if e['rtt'] == rtt and e['ablation'] == abl]
            speeds = {}
            for e in exps:
                speeds[e['model_label']] = e['speeds'].get('wan', 0)

            models = ['0.5B → 1.5B', '1.5B → 7B']
            vals = [speeds.get(m, 0) for m in models]
            colors = ['#3498db', '#e74c3c']
            bars = ax.bar(models, vals, color=colors, edgecolor='black', width=0.5)
            ax.set_ylabel('WAN Speed (tok/s)')
            ax.set_title(f'{ablation_labels[abl]} @ RTT={rtt}ms')
            ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_mode_speedup(experiments, output_dir):
    """Plot speedup of speculative_decoding and wan vs autoregressive baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ablation_colors = {
        'full': '#2ecc71',
        'no_adaptive': '#e74c3c',
        'no_fallback': '#3498db',
        'no_compression': '#9b59b6',
    }
    ablation_labels = {
        'full': 'Full',
        'no_adaptive': 'No Adaptive',
        'no_fallback': 'No Fallback',
        'no_compression': 'No Compression',
    }

    for idx, model_label in enumerate(['0.5B → 1.5B', '1.5B → 7B']):
        ax = axes[idx]
        model_exps = [e for e in experiments if e['model_label'] == model_label]
        for abl in ['full', 'no_adaptive', 'no_fallback', 'no_compression']:
            points_sd = []
            points_wan = []
            for e in model_exps:
                if e['ablation'] != abl:
                    continue
                ar = e['speeds'].get('autoregressive', 0)
                sd = e['speeds'].get('speculative_decoding', 0)
                wan = e['speeds'].get('wan', 0)
                if ar > 0:
                    points_sd.append((e['rtt'], sd / ar))
                    points_wan.append((e['rtt'], wan / ar))
            if points_sd:
                points_sd = sorted(points_sd, key=lambda x: x[0])
                x_sd, y_sd = zip(*points_sd)
                ax.plot(x_sd, y_sd, marker='s', linewidth=2, markersize=7,
                       linestyle='--', label=f'{ablation_labels[abl]} (SD)',
                       color=ablation_colors[abl], alpha=0.7)
            if points_wan:
                points_wan = sorted(points_wan, key=lambda x: x[0])
                x_wan, y_wan = zip(*points_wan)
                ax.plot(x_wan, y_wan, marker='o', linewidth=2, markersize=7,
                       label=f'{ablation_labels[abl]} (WAN)',
                       color=ablation_colors[abl])
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5)
        ax.set_xlabel('RTT (ms)')
        ax.set_ylabel('Speedup vs Autoregressive')
        ax.set_title(f'Speedup Analysis ({model_label})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'mode_speedup.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_benchmark_results(exp_dir, output_dir):
    """Plot actual benchmark results if available."""
    benchmarks = []
    for prefix in ['benchmark_humaneval', 'benchmark_gsm8k', 'benchmark_mgsm']:
        for item in os.listdir(exp_dir):
            if item.startswith(prefix) and os.path.isdir(os.path.join(exp_dir, item)):
                result_file = os.path.join(exp_dir, item, f"eval_{prefix.split('_')[1]}_results.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        benchmarks.append((item, json.load(f)))

    if not benchmarks:
        print("No benchmark results found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'autoregressive': '#3498db', 'speculative_decoding': '#2ecc71', 'wan': '#e74c3c'}
    x_labels = []
    x_pos = []
    width = 0.25
    idx = 0

    for name, data in benchmarks:
        for run in data.get('runs', []):
            mode = run['mode']
            speed = run.get('avg_speed', 0)
            label = f"{name}\n{mode}"
            if label not in x_labels:
                x_labels.append(label)
                x_pos.append(idx)
                idx += 1

    # Simplified: just plot speed per mode per benchmark
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6*len(benchmarks), 5))
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, benchmarks):
        modes = [r['mode'] for r in data.get('runs', [])]
        speeds = [r.get('avg_speed', 0) for r in data.get('runs', [])]
        bar_colors = [colors.get(m, '#9b59b6') for m in modes]
        bars = ax.bar(modes, speeds, color=bar_colors, edgecolor='black')
        ax.set_ylabel('Speed (tok/s)')
        ax.set_title(name.replace('_', ' '))
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'benchmark_speeds.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    experiments = discover_ablation_experiments(args.exp_dir)
    print(f"Discovered {len(experiments)} ablation experiments")

    if experiments:
        plot_rtt_sweep(experiments, args.output_dir)
        plot_ablation_comparison(experiments, args.output_dir)
        plot_model_comparison(experiments, args.output_dir)
        plot_mode_speedup(experiments, args.output_dir)
    else:
        print("No ablation experiments found.")

    plot_benchmark_results(args.exp_dir, args.output_dir)
    print(f"\nAll ablation plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
