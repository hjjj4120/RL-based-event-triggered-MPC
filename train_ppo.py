import argparse
import datetime
import json
import os
import random
import csv

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.io import savemat

from veh_env2 import vehEnv, DEFAULT_P
from PPO.config import AgentConfig
from PPO.network import MlpPolicy

device = torch.device("cpu")


class Agent(AgentConfig):
    def __init__(self, dir, writer, args):
        self.args = args
        self.env = vehEnv(
            T=20,
            rho=args.rho,
            sph=args.sph,
            np_eco=args.np_eco,
            np_pro=args.np_pro,
            path_set=args.path_set,
            curriculum=args.curriculum,
            domain_rand=args.domain_rand,
            seed=args.seed,
            difficulty=args.difficulty,
            total_episodes=args.episodes,
        )
        self.env_test = vehEnv(
            T=20,
            rho=args.rho,
            sph=args.sph,
            np_eco=args.np_eco,
            np_pro=args.np_pro,
            path_set=args.path_set,
            curriculum=0,
            domain_rand=0,
            seed=args.seed + 100,
            difficulty=args.difficulty,
            total_episodes=args.episodes,
        )
        self.action_size = self.env.action_space
        self.policy_network = MlpPolicy(action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.loss = 0
        self.writer = writer
        self.dir = dir
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([])}

        self.csv_path = os.path.join(dir, 'train_logs.csv')
        self.summary_path = os.path.join(dir, 'summary.json')
        self._ensure_log_header()
        self.best_eval_return = -float("inf")
        self.checkpoint_dir = os.path.join(dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        episode = 0
        step = 0

        # A new episode
        for _ in range(self.args.episodes):
            start_step = step
            episode += 1
            episode_length = 0
            terminal = False

            # Get initial state
            state = self.env.reset(episode_idx=episode - 1)
            current_state = state
            total_episode_reward = 0
            actions = []
            solve_times = []
            solve_ok_list = []

            # do one episode
            while not terminal:
                step += 1
                episode_length += 1

                # choose an action
                prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
                action = torch.distributions.Categorical(prob_a).sample().item()
                actions.append(action)

                # act
                state, reward, terminal, info = self.env.step(action)
                new_state = state
                solve_times.append(info["solve_time_ms"])
                solve_ok_list.append(1 if info["solve_ok"] else 0)

                self.add_memory(current_state, action, reward, new_state, terminal, prob_a[action].item())

                current_state = new_state
                total_episode_reward += reward

                if terminal:
                    episode_length = step - start_step

                    # add logs
                    self.writer.add_scalar('Train/steps', episode_length, episode)
                    self.writer.add_scalar('Train/EpisodeReturns', total_episode_reward, episode)
                    action_ratio = sum(actions) / len(actions)
                    self.writer.add_scalar('Train/action_ratio', action_ratio, episode)
                    if self.args.sph:
                        freq_hold, freq_eco, freq_pro = self._action_frequencies(actions)
                        self.writer.add_scalar('Train/freq_hold', freq_hold, episode)
                        self.writer.add_scalar('Train/freq_eco', freq_eco, episode)
                        self.writer.add_scalar('Train/freq_pro', freq_pro, episode)

                    self.finish_path(episode_length)

                    # do a test
                    if episode % self.eval_freq == 0:
                        state_test = self.env_test.reset(episode_idx=episode - 1)
                        terminal_test = False
                        test_steps = 0
                        rewards_test = 0
                        actions_test = []
                        solve_times_test = []
                        solve_ok_test = []

                        # save logs
                        x = [state_test[0]]
                        rl = [state_test[2]]  # rl output
                        gt = [self.env_test.reference(state_test[0])]  # ground truth
                        act = []
                        jmpcs = []

                        while not terminal_test:
                            test_steps += 1
                            prob_a_test = self.policy_network.pi(torch.FloatTensor(state_test).to(device))
                            action_test = torch.distributions.Categorical(prob_a_test).sample().item()
                            state_test, reward_test, terminal_test, info = self.env_test.step(action_test)
                            rewards_test += reward_test
                            actions_test.append(action_test)
                            solve_times_test.append(info["solve_time_ms"])
                            solve_ok_test.append(1 if info["solve_ok"] else 0)

                            act.append(action_test)
                            x.append(state_test[0])
                            rl.append(state_test[2])
                            gt.append(self.env_test.reference(state_test[0]))
                            jmpcs.append(info["jmpc"])

                        self.writer.add_scalar('Eval/steps', test_steps, episode)
                        self.writer.add_scalar('Eval/EpisodeReturns', rewards_test, episode)
                        eval_action_ratio = sum(actions_test) / len(actions_test)
                        self.writer.add_scalar('Eval/action_ratio', eval_action_ratio, episode)
                        if self.args.sph:
                            freq_hold, freq_eco, freq_pro = self._action_frequencies(actions_test)
                            self.writer.add_scalar('Eval/freq_hold', freq_hold, episode)
                            self.writer.add_scalar('Eval/freq_eco', freq_eco, episode)
                            self.writer.add_scalar('Eval/freq_pro', freq_pro, episode)

                        save_mat = {
                            'act': act,
                            'x': x,
                            'rl': rl,
                            'gt': gt,
                            'jmpcs': jmpcs}
                        if not os.path.exists(self.dir + '/results'):
                            os.mkdir(self.dir + '/results')

                        savemat(self.dir + '/results/ppo_results_{}.mat'.format(episode), save_mat)

                        max_step = len(rl)
                        eval_error = np.array(rl[:max_step]) - np.array(gt[:max_step])
                        solve_stats = self._solve_stats(solve_times_test, solve_ok_test)
                        row = [
                            'PPO',
                            episode,
                            max_step,
                            '{:.3f}'.format(total_episode_reward),
                            '{:.3f}'.format(rewards_test),
                            '{:.3f}'.format(sum(jmpcs)),
                            '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                            '{:.3f}'.format(np.min(eval_error[15:]) if len(eval_error) > 15 else np.min(eval_error)),
                            '{:.3f}'.format(np.max(eval_error[15:]) if len(eval_error) > 15 else np.max(eval_error)),
                            '{:.3f}'.format(eval_action_ratio),
                            '{:.3f}'.format(solve_stats["mean_solve_time_ms"]),
                            '{:.3f}'.format(solve_stats["p95_solve_time_ms"]),
                            '{:.3f}'.format(solve_stats["solve_fail_rate"]),
                        ]
                        if self.args.sph:
                            freq_hold, freq_eco, freq_pro = self._action_frequencies(actions_test)
                            row.extend(['{:.3f}'.format(freq_hold), '{:.3f}'.format(freq_eco), '{:.3f}'.format(freq_pro)])

                        with open(self.csv_path, 'a+', newline='') as write_obj:
                            csv_writer = csv.writer(write_obj)
                            csv_writer.writerow(row)

                        self._save_summary(episode, rewards_test, solve_stats, eval_action_ratio, actions_test)
                        if rewards_test > self.best_eval_return:
                            self.best_eval_return = rewards_test
                            self._save_checkpoint(episode, best=True)

                        if self.args.eval_suite:
                            from experiments.eval_suite import run_eval_suite
                            run_eval_suite(
                                run_dir=self.dir,
                                seed=self.args.seed,
                                path_set=self.args.path_set,
                                difficulty=self.args.difficulty,
                                episodes=3,
                                max_step=100,
                                sph=self.args.sph,
                                np_eco=self.args.np_eco,
                                np_pro=self.args.np_pro,
                            )

                    self.env.reset()
                    break

            if episode % self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()

        self.env.close()

    def _action_frequencies(self, actions):
        total = max(1, len(actions))
        freq_hold = actions.count(0) / total
        freq_eco = actions.count(1) / total
        freq_pro = actions.count(2) / total
        return freq_hold, freq_eco, freq_pro

    def _solve_stats(self, times_ms, ok_list):
        if not times_ms:
            return {"mean_solve_time_ms": 0.0, "p95_solve_time_ms": 0.0, "solve_fail_rate": 0.0}
        mean_time = float(np.mean(times_ms))
        p95_time = float(np.percentile(times_ms, 95))
        fail_rate = 1.0 - float(np.mean(ok_list)) if ok_list else 0.0
        return {
            "mean_solve_time_ms": mean_time,
            "p95_solve_time_ms": p95_time,
            "solve_fail_rate": fail_rate,
        }

    def _save_checkpoint(self, episode, best=False):
        tag = "best" if best else f"episode_{episode}"
        path = os.path.join(self.checkpoint_dir, f"ppo_{tag}.pt")
        torch.save(self.policy_network.state_dict(), path)

    def _ensure_log_header(self):
        if os.path.isfile(self.csv_path):
            return
        fieldnames = [
            'Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
            'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq',
            'mean_solve_time_ms', 'p95_solve_time_ms', 'solve_fail_rate',
        ]
        if self.args.sph:
            fieldnames.extend(['freq_hold', 'freq_eco', 'freq_pro'])
        with open(self.csv_path, mode='w') as csv_file:
            writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_csv.writeheader()

    def _save_summary(self, episode, eval_return, solve_stats, action_ratio, actions):
        summary = {
            "episode": episode,
            "eval_return": eval_return,
            "action_ratio": action_ratio,
            "mean_solve_time_ms": solve_stats["mean_solve_time_ms"],
            "p95_solve_time_ms": solve_stats["p95_solve_time_ms"],
            "solve_fail_rate": solve_stats["solve_fail_rate"],
        }
        if self.args.sph:
            freq_hold, freq_eco, freq_pro = self._action_frequencies(actions)
            summary.update(
                {"freq_hold": freq_hold, "freq_eco": freq_eco, "freq_pro": freq_pro}
            )
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']))
        old_probs_a = torch.FloatTensor(self.memory['action_prob'])
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory['advantage'])
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(self.memory['advantage'])
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        v_loss = 0.5 * (pred_v - self.memory['td_target']).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)

    def finish_path(self, length):
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]

        td_target = torch.FloatTensor(reward) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(next_state)) * (torch.FloatTensor(terminal))
        delta = td_target - self.policy_network.v(torch.FloatTensor(state))
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.data), dim=0)
        self.memory['advantage'] += advantages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--sph', type=int, default=0)
    parser.add_argument('--np_eco', type=int, default=DEFAULT_P)
    parser.add_argument('--np_pro', type=int, default=DEFAULT_P)
    parser.add_argument('--path_set', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--domain_rand', type=int, default=0)
    parser.add_argument('--eval_suite', type=int, default=0)
    parser.add_argument('--difficulty', type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    global device
    device = torch.device("cpu")
    if args.run_dir is None:
        dir_name = os.path.join(
            'runs',
            'PPO' + '_rho_' + str(args.rho) + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
    else:
        dir_name = args.run_dir
    os.makedirs(dir_name, exist_ok=True)
    writer = SummaryWriter(log_dir=dir_name)

    agent = Agent(dir_name, writer, args)
    agent.train()
