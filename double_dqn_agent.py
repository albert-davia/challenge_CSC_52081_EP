"""
Double DQN Agent - Turbofan Engine Maintenance Challenge

Architecture
------------
  Input  : flatten the (10, 9) observation window -> 90 raw features
  Network: MLP  90 -> 128 -> 64 -> 3 Q-values (one per action)
  Double DQN:
    - Online net  selects the best next action  (argmax Q_online)
    - Target net  evaluates that action's value  (Q_target(s', a*))
    - TD target   r + gamma * Q_target(s', argmax_a Q_online(s', a))
  Replay buffer: circular, capacity 50k transitions
  Exploration  : epsilon-greedy, decay every episode

Run
---
  python double_dqn_agent.py           # train from scratch
  python double_dqn_agent.py --eval    # greedy eval with saved model
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

load_dotenv()

from student_client import create_student_gym_env, get_leaderboard_score

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
GAMMA            = 0.99      # discount factor
LR               = 3e-4      # Adam learning rate
BATCH_SIZE       = 64        # replay batch size
REPLAY_CAPACITY  = 50000    # max transitions stored
EPSILON_START    = 1.0       # initial exploration rate
EPSILON_MIN      = 0.05      # floor for epsilon
EPSILON_DECAY    = 0.995     # multiply epsilon after each episode
TARGET_UPDATE    = 10        # hard-update target net every N episodes
MIN_REPLAY       = 1_000     # wait for this many transitions before training

TRAIN_EPISODES   = 500       # total training episodes
EVAL_EVERY       = 25        # run a greedy evaluation every N episodes
EVAL_EPISODES    = 5         # number of greedy episodes per evaluation

MODEL_PATH       = "double_dqn_model.pt"   # where to save/load weights

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------
OBS_WINDOW  = 10   # flights per step
N_SENSORS   = 9
OBS_DIM     = OBS_WINDOW * N_SENSORS   # 90 input features
N_ACTIONS   = 3
HIDDEN_1    = 128
HIDDEN_2    = 64

# Approximate per-sensor means and stds from explore_trajectories.ipynb
# Used to normalise inputs so all features are O(1) for the network.
SENSOR_MEAN = np.array([793.1, 19315.4, 335.1, 1118.3, 0.372,
                         1358857.5, 3953.9, 0.0, 9.44], dtype=np.float32)
SENSOR_STD  = np.array([10.0,  300.0,   1.5,   5.0,   0.002,
                         25000.0,     5.0,   1.0,  0.5],  dtype=np.float32)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def obs_to_features(obs: np.ndarray) -> np.ndarray:
    """
    Convert a raw observation to a normalised flat (90,) feature vector.

    Handles both shapes returned by the environment:
      - (9,)   from env.reset()
      - (10,9) from env.step()
    """
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 1:           # (9,) from reset — pad with copies to fill 10 rows
        obs = np.tile(obs, (OBS_WINDOW, 1))          # (10, 9)
    elif obs.shape[0] < OBS_WINDOW:
        pad = np.tile(obs[0], (OBS_WINDOW - obs.shape[0], 1))
        obs = np.vstack([pad, obs])                   # (10, 9)
    elif obs.shape[0] > OBS_WINDOW:
        obs = obs[-OBS_WINDOW:]                       # keep most recent 10

    # Per-sensor z-score normalisation
    normalised = (obs - SENSOR_MEAN) / (SENSOR_STD + 1e-8)   # (10, 9)
    return normalised.flatten()                               # (90,)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Double DQN Agent
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    def __init__(self):
        self.device = torch.device("cpu")   # CPU is fine for this small net

        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self._hard_update_target()            # sync weights at init

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — more stable than MSE

        self.replay  = ReplayBuffer(REPLAY_CAPACITY)
        self.epsilon = EPSILON_START

        self.steps_trained = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(s)
            return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training step (Double DQN update)
    # ------------------------------------------------------------------

    def train_step(self):
        if len(self.replay) < MIN_REPLAY:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        states      = states.to(self.device)
        next_states = next_states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        dones       = dones.to(self.device)

        # ── Current Q-values ───────────────────────────────────────────
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Double DQN target ──────────────────────────────────────────
        with torch.no_grad():
            # Online net selects best action in next state
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates that action's value
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q     = rewards + GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_trained += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_PATH):
        torch.save({
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_trained': self.steps_trained,
        }, path)
        print(f"  Model saved -> {path}")

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            print(f"  No saved model found at {path}, starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_state_dict'])
        self.target_net.load_state_dict(ckpt['target_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epsilon      = ckpt.get('epsilon', EPSILON_MIN)
        self.steps_trained = ckpt.get('steps_trained', 0)
        print(f"  Model loaded from {path}  (epsilon={self.epsilon:.3f}, steps={self.steps_trained})")


# ---------------------------------------------------------------------------
# Episode runner helpers
# ---------------------------------------------------------------------------

def run_training_episode(env, agent: DoubleDQNAgent):
    """Play one episode, push transitions into replay, run train steps."""
    obs, _ = env.reset()
    state  = obs_to_features(obs)

    total_reward = 0.0
    losses       = []
    n_repairs    = 0

    for _ in range(400):   # generous cap
        action     = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = obs_to_features(obs)
        done       = terminated or truncated

        agent.replay.push(state, action, reward, next_state, float(done))

        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        total_reward += reward
        state         = next_state
        if action == 1:
            n_repairs += 1

        if done:
            break

    return total_reward, np.mean(losses) if losses else 0.0, n_repairs


def run_eval_episode(env, agent: DoubleDQNAgent):
    """Play one greedy episode (no exploration)."""
    obs, _ = env.reset()
    state  = obs_to_features(obs)

    total_reward = 0.0

    for _ in range(400):
        action = agent.select_action(state, greedy=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = obs_to_features(obs)
        if terminated or truncated:
            break

    return total_reward


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    env   = create_student_gym_env()
    agent = DoubleDQNAgent()

    # Resume from checkpoint if one exists
    if os.path.exists(MODEL_PATH):
        agent.load()

    best_eval_avg = -float("inf")
    all_train_rewards = []

    print(f"\nDouble DQN Training - {TRAIN_EPISODES} episodes")
    print(f"  OBS_DIM={OBS_DIM}, HIDDEN={HIDDEN_1}/{HIDDEN_2}, LR={LR}, GAMMA={GAMMA}")
    print(f"  epsilon_start={EPSILON_START}, epsilon_min={EPSILON_MIN}, decay={EPSILON_DECAY}")
    print("-" * 75)

    for ep in range(1, TRAIN_EPISODES + 1):

        total_r, avg_loss, n_repairs = run_training_episode(env, agent)
        all_train_rewards.append(total_r)
        agent.decay_epsilon()

        # Hard-update target network periodically
        if ep % TARGET_UPDATE == 0:
            agent._hard_update_target()

        print(
            f"Ep {ep:>4d}/{TRAIN_EPISODES} | "
            f"reward={total_r:>8.1f} | "
            f"loss={avg_loss:>7.4f} | "
            f"eps={agent.epsilon:.3f} | "
            f"repairs={n_repairs:>2d} | "
            f"buf={len(agent.replay):>6d}",
            flush=True,
        )

        # ── Periodic evaluation ────────────────────────────────────────
        if ep % EVAL_EVERY == 0:
            eval_rewards = [run_eval_episode(env, agent) for _ in range(EVAL_EPISODES)]
            eval_avg = np.mean(eval_rewards)
            last_100_avg = np.mean(all_train_rewards[-100:]) if len(all_train_rewards) >= 100 else np.mean(all_train_rewards)

            print(f"\n  == Eval @ ep {ep} ==")
            print(f"     Greedy avg ({EVAL_EPISODES} eps): {eval_avg:.1f}   best: {max(eval_rewards):.1f}")
            print(f"     Train avg (last 100):        {last_100_avg:.1f}   (Baseline 1: 3951)")
            print(f"     Replay buffer:               {len(agent.replay)}")
            print()

            if eval_avg > best_eval_avg:
                best_eval_avg = eval_avg
                agent.save(MODEL_PATH)
                print(f"  *** New best eval avg: {best_eval_avg:.1f} — model saved ***\n")

    env.close()

    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)
    last_100 = all_train_rewards[-100:]
    print(f"  Train avg (last 100): {np.mean(last_100):.1f}")
    print(f"  Best eval avg:        {best_eval_avg:.1f}")
    print()

    # Ask if we should run the leaderboard scoring with the trained model
    print("Running 120 greedy episodes for leaderboard scoring...")
    score_env    = create_student_gym_env()
    agent.load()   # load best checkpoint
    agent.epsilon = 0.0   # fully greedy

    score_rewards = []
    for ep in range(1, 121):
        r = run_eval_episode(score_env, agent)
        score_rewards.append(r)
        print(f"  Score ep {ep:>3d}/120 | reward={r:>8.1f}", flush=True)

    score_env.close()

    print("\n" + "=" * 75)
    print("LEADERBOARD SCORING SUMMARY")
    print("=" * 75)
    print(f"  Total reward (120 eps):    {sum(score_rewards):.1f}")
    print(f"  Average reward:            {np.mean(score_rewards):.1f}  (Baseline 1: 3951)")
    print(f"  Best episode:              {max(score_rewards):.1f}")
    print()

    user_token = os.getenv("USER_TOKEN", "student_user")
    try:
        df = get_leaderboard_score(user_token=user_token)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not fetch leaderboard score: {e}")


# ---------------------------------------------------------------------------
# Evaluation-only mode
# ---------------------------------------------------------------------------

def evaluate():
    """Load saved model and run 120 greedy episodes for the leaderboard."""
    env   = create_student_gym_env()
    agent = DoubleDQNAgent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0   # fully greedy

    rewards = []
    for ep in range(1, 121):
        r = run_eval_episode(env, agent)
        rewards.append(r)
        print(f"Ep {ep:>3d}/120 | reward={r:>8.1f}", flush=True)

    env.close()
    print("\n" + "=" * 60)
    print(f"  Average reward: {np.mean(rewards):.1f}  (Baseline 1: 3951)")
    print(f"  Total reward:   {sum(rewards):.1f}")
    print("=" * 60)

    user_token = os.getenv("USER_TOKEN", "student_user")
    try:
        df = get_leaderboard_score(user_token=user_token)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not fetch leaderboard score: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Load saved model and run greedy evaluation only.")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train()
