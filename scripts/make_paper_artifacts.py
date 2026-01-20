import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def find_files(root, filename):
    matches = []
    for dirpath, _, files in os.walk(root):
        if filename in files:
            matches.append(os.path.join(dirpath, filename))
    return matches


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_eval_suite(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["sph"], row["path_set"])
        groups[key].append(row)

    table_rows = []
    for key, items in groups.items():
        metrics = {
            "return": [to_float(r["return"]) for r in items],
            "tracking_rmse": [to_float(r["tracking_rmse"]) for r in items],
            "mean_solve_time_ms": [to_float(r["mean_solve_time_ms"]) for r in items],
            "solve_fail_rate": [to_float(r["solve_fail_rate"]) for r in items],
        }
        row = {
            "method": key[0],
            "sph": key[1],
            "path_set": key[2],
        }
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        table_rows.append(row)
    return table_rows


def write_table(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curve(train_logs, out_path):
    plt.figure(figsize=(6, 4))
    for label, rows in train_logs.items():
        episodes = [int(r["Epochs"]) for r in rows]
        eval_returns = [to_float(r["Evaluation Return"]) for r in rows]
        if episodes:
            plt.plot(episodes, eval_returns, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Eval Return")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_action_freq(eval_rows, out_path):
    plt.figure(figsize=(6, 4))
    labels = []
    hold_vals = []
    eco_vals = []
    pro_vals = []
    groups = defaultdict(list)
    for row in eval_rows:
        key = f"{row['method']}-sph{row['sph']}-{row['path_set']}"
        groups[key].append(row)
    for label, items in groups.items():
        labels.append(label)
        hold_vals.append(np.mean([to_float(i["freq_hold"]) for i in items]))
        eco_vals.append(np.mean([to_float(i["freq_eco"]) for i in items]))
        pro_vals.append(np.mean([to_float(i["freq_pro"]) for i in items]))
    x = np.arange(len(labels))
    width = 0.25
    plt.bar(x - width, hold_vals, width, label="hold")
    plt.bar(x, eco_vals, width, label="eco")
    plt.bar(x + width, pro_vals, width, label="pro")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Action Frequency")
    plt.title("Action Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_solve_time(eval_rows, out_path):
    plt.figure(figsize=(6, 4))
    labels = []
    means = []
    p95s = []
    groups = defaultdict(list)
    for row in eval_rows:
        key = f"{row['method']}-sph{row['sph']}-{row['path_set']}"
        groups[key].append(row)
    for label, items in groups.items():
        labels.append(label)
        means.append(np.mean([to_float(i["mean_solve_time_ms"]) for i in items]))
        p95s.append(np.mean([to_float(i["p95_solve_time_ms"]) for i in items]))
    x = np.arange(len(labels))
    plt.plot(x, means, marker="o", label="mean")
    plt.plot(x, p95s, marker="o", label="p95")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Solve Time (ms)")
    plt.title("Solve Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(results_root, out_dir):
    eval_paths = find_files(results_root, 'eval_suite.csv')
    eval_rows = []
    for path in eval_paths:
        eval_rows.extend(load_csv(path))

    if not eval_rows:
        raise RuntimeError(f"No eval_suite.csv files found under {results_root}")

    table_rows = summarize_eval_suite(eval_rows)
    table_path = os.path.join(out_dir, 'table_main.csv')
    write_table(table_path, table_rows)

    train_log_paths = find_files(results_root, 'train_logs.csv')
    train_logs = {}
    for path in train_log_paths:
        label = os.path.basename(os.path.dirname(path))
        train_logs[label] = load_csv(path)

    os.makedirs(out_dir, exist_ok=True)
    plot_learning_curve(train_logs, os.path.join(out_dir, 'fig_learning_curve.png'))
    plot_action_freq(eval_rows, os.path.join(out_dir, 'fig_action_freq.png'))
    plot_solve_time(eval_rows, os.path.join(out_dir, 'fig_solve_time.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='paper_artifacts')
    args = parser.parse_args()
    main(args.results_root, args.out_dir)
