"""
Heuristic Agent - Turbofan Engine Maintenance Challenge

Policy:
  - Observe the 10-flight window returned by each step()
  - Compute HPC_Tout window mean and trend (slope)
  - If mean > SELL_THRESHOLD     → sell  (action 2)
  - If mean > REPAIR_THRESHOLD
       AND trend > TREND_THRESHOLD → repair (action 1)
  - Otherwise                    → do nothing (action 0)

HOW TO USE:
  1. Run explore_trajectories.ipynb first to pick good threshold values.
  2. Fill in the three constants below.
  3. Run:  python heuristic_agent.py
  4. It will play NUM_EPISODES episodes and print a live leaderboard score.
"""

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from student_client import create_student_gym_env, get_leaderboard_score
import os

# ─────────────────────────────────────────────────────────────
#  THRESHOLDS  (HPC_Tout DECREASES with degradation — lower = worse)
#
#  From explore_trajectories.ipynb:
#    Healthy mean  : ~793.1
#    Failing mean  : ~784.5
#    Failing 90th pct: 786.9   → below this = clearly failing
#
#  Logic: sell when mean is too LOW; repair when trending downward.
# ─────────────────────────────────────────────────────────────
SELL_THRESHOLD   = 787.0  # HPC_Tout mean BELOW this → sell
REPAIR_THRESHOLD = 790.5  # HPC_Tout mean BELOW this → consider repair
TREND_THRESHOLD  = -1.5   # HPC_Tout trend BELOW this (negative slope) → repair

# How many episodes to run for scoring (last 100 count on the leaderboard)
NUM_EPISODES = 120

# Print leaderboard score every N episodes
LEADERBOARD_EVERY = 20
# ─────────────────────────────────────────────────────────────

SENSOR_NAMES = ['HPC_Tout', 'HP_Nmech', 'HPC_Tin', 'LPT_Tin',
                'Fuel_flow', 'HPC_Pout_st', 'LP_Nmech', 'phase_type', 'DTAMB']


def heuristic_action(obs_window: np.ndarray) -> int:
    """
    Decide an action from a (10, 9) observation window.

    HPC_Tout (col 0) DECREASES as the engine degrades.
      < SELL_THRESHOLD   → engine too worn → sell
      < REPAIR_THRESHOLD AND trend < TREND_THRESHOLD → degrading actively → repair
      otherwise → do nothing

    Args:
        obs_window: numpy array of shape (10, 9)

    Returns:
        0 = do nothing, 1 = repair, 2 = sell
    """
    # reset() returns shape (9,); step() returns shape (10, 9)
    if obs_window.ndim == 1:
        obs_window = obs_window.reshape(1, -1)

    hpc_tout = obs_window[:, 0]
    window_mean  = hpc_tout.mean()
    window_trend = hpc_tout[-1] - hpc_tout[0]  # negative = degrading

    if window_mean < SELL_THRESHOLD:
        return 2  # engine too degraded — sell

    if window_mean < REPAIR_THRESHOLD and window_trend < TREND_THRESHOLD:
        return 1  # degradation active and worsening — repair

    return 0  # engine still healthy — keep running


def run_episode(env) -> dict:
    """Run one episode with the heuristic policy. Returns a summary dict."""
    obs, info = env.reset()

    total_reward = 0.0
    n_repairs    = 0
    n_steps      = 0
    terminated   = False
    truncated    = False

    for _ in range(300):  # safety cap
        action = heuristic_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        n_steps      += 1
        if action == 1:
            n_repairs += 1

        if terminated or truncated:
            break

    failure = terminated and info.get('message', '').lower().find('fail') >= 0

    return {
        'total_reward': total_reward,
        'n_steps':      n_steps,
        'n_repairs':    n_repairs,
        'failure':      failure,
    }


def validate_thresholds():
    if REPAIR_THRESHOLD <= SELL_THRESHOLD:
        print("WARNING: REPAIR_THRESHOLD should be greater than SELL_THRESHOLD "
              "(since lower HPC_Tout = worse — repair zone is above sell zone).")


def main():
    validate_thresholds()

    user_token = os.getenv('USER_TOKEN', 'student_user')
    env = create_student_gym_env()

    print(f"\nHeuristic Agent - {NUM_EPISODES} episodes")
    print(f"  SELL_THRESHOLD   = {SELL_THRESHOLD}")
    print(f"  REPAIR_THRESHOLD = {REPAIR_THRESHOLD}")
    print(f"  TREND_THRESHOLD  = {TREND_THRESHOLD}")
    print("-" * 60)

    all_rewards = []
    n_failures  = 0

    for ep in range(1, NUM_EPISODES + 1):
        result = run_episode(env)
        all_rewards.append(result['total_reward'])
        if result['failure']:
            n_failures += 1

        print(
            f"Ep {ep:>4d}/{NUM_EPISODES} | "
            f"reward={result['total_reward']:>8.1f} | "
            f"steps={result['n_steps']:>4d} | "
            f"repairs={result['n_repairs']:>3d} | "
            f"{'FAIL' if result['failure'] else '    '}"
        )

        # Print running summary every N episodes
        if ep % LEADERBOARD_EVERY == 0:
            last_100 = all_rewards[-100:]
            print(f"\n  -- Summary after {ep} episodes --")
            print(f"     Total reward (last {len(last_100)}): {sum(last_100):.1f}")
            print(f"     Avg reward:    {np.mean(last_100):.1f}  (Baseline 1: 3951)")
            print(f"     Best reward:   {max(last_100):.1f}")
            print(f"     Failure rate:  {n_failures / ep:.1%}")
            print()

    env.close()

    # Final summary
    last_100 = all_rewards[-100:]
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Episodes run:              {NUM_EPISODES}")
    print(f"  Total reward (last 100):   {sum(last_100):.1f}  (Baseline 1: 395144)")
    print(f"  Average reward (last 100): {np.mean(last_100):.1f}  (Baseline 1: 3951)")
    print(f"  Best reward:               {max(last_100):.1f}")
    print(f"  Failure rate:              {n_failures / NUM_EPISODES:.1%}  (Baseline 1: 5%)")
    print()

    # Fetch leaderboard score
    print("Fetching leaderboard score from server...")
    try:
        df = get_leaderboard_score(user_token=user_token)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not fetch leaderboard score: {e}")


if __name__ == "__main__":
    main()
