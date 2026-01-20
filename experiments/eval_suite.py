import argparse
import csv
import json
import os
import numpy as np
import torch

from PPO.network import MlpPolicy
from veh_env2 import vehEnv, DEFAULT_P


def load_policy(run_dir, action_size):
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    best_path = os.path.join(checkpoint_dir, 'ppo_best.pt')
    model = MlpPolicy(action_size=action_size)
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location='cpu'))
        model.eval()
        return model
    if os.path.isdir(checkpoint_dir):
        candidates = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if candidates:
            latest = sorted(candidates)[-1]
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest), map_location='cpu'))
            model.eval()
            return model
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")


def run_eval_suite(
    run_dir,
    seed,
    path_set,
    difficulty,
    episodes,
    max_step,
    sph,
    np_eco,
    np_pro,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = vehEnv(
        T=20,
        rho=0,
        sph=sph,
        np_eco=np_eco,
        np_pro=np_pro,
        path_set=path_set,
        curriculum=0,
        domain_rand=0,
        seed=seed,
        difficulty=difficulty,
        total_episodes=episodes,
    )
    policy = load_policy(run_dir, env.action_space)
    rows = []
    for ep in range(episodes):
        state = env.reset(episode_idx=ep)
        done = False
        step = 0
        episode_return = 0.0
        jmpc_sum = 0.0
        actions = []
        solve_times = []
        solve_ok = []
        errors = []

        while not done and step < max_step:
            with torch.no_grad():
                prob_a = policy.pi(torch.FloatTensor(state))
                action = prob_a.argmax().item()
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            jmpc_sum += info["jmpc"]
            actions.append(action)
            solve_times.append(info["solve_time_ms"])
            solve_ok.append(1 if info["solve_ok"] else 0)
            errors.append(next_state[2] - env.reference(next_state[0]))
            state = next_state
            step += 1

        tracking_rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        freq_hold = actions.count(0) / max(1, len(actions))
        freq_eco = actions.count(1) / max(1, len(actions))
        freq_pro = actions.count(2) / max(1, len(actions))
        solve_fail_rate = 1.0 - float(np.mean(solve_ok)) if solve_ok else 0.0
        rows.append(
            {
                "method": "PPO",
                "sph": sph,
                "path_set": path_set,
                "difficulty": difficulty,
                "seed": seed,
                "episode": ep + 1,
                "return": episode_return,
                "jmpc": jmpc_sum,
                "tracking_rmse": tracking_rmse,
                "action_freq": float(np.mean(actions)) if actions else 0.0,
                "freq_hold": freq_hold,
                "freq_eco": freq_eco,
                "freq_pro": freq_pro,
                "mean_solve_time_ms": float(np.mean(solve_times)) if solve_times else 0.0,
                "p95_solve_time_ms": float(np.percentile(solve_times, 95)) if solve_times else 0.0,
                "solve_fail_rate": solve_fail_rate,
            }
        )

    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, 'eval_suite.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for key in ["return", "jmpc", "tracking_rmse", "mean_solve_time_ms", "p95_solve_time_ms", "solve_fail_rate"]:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    json_path = os.path.join(run_dir, 'eval_suite.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path_set', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--difficulty', type=float, default=1.0)
    parser.add_argument('--max_step', type=int, default=100)
    parser.add_argument('--sph', type=int, default=0)
    parser.add_argument('--np_eco', type=int, default=DEFAULT_P)
    parser.add_argument('--np_pro', type=int, default=DEFAULT_P)
    args = parser.parse_args()
    run_eval_suite(
        run_dir=args.run_dir,
        seed=args.seed,
        path_set=args.path_set,
        difficulty=args.difficulty,
        episodes=args.episodes,
        max_step=args.max_step,
        sph=args.sph,
        np_eco=args.np_eco,
        np_pro=args.np_pro,
    )
