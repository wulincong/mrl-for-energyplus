import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from gym_energyplus.envs.energyplus_multiagent_env import EnergyPlusMultiAgentEnv


def run_rollout(env, zone_id: str, n_steps: int):
    obs = env.reset()
    comps = {"gauss": [], "trap": [], "power_pen": [], "smooth": [], "total": []}
    for _ in range(n_steps):
        action_dict = {aid: env.action_space.sample() for aid in env.AGENT_IDS}
        obs, rew_dict, done_dict, info_dict = env.step(action_dict)
        comp = info_dict[zone_id]["reward_components"]
        for k in comps:
            comps[k].append(comp[k])
        if done_dict.get("__all__", False):
            break
    return comps


def save_csv(out_path: str, comps):
    keys = list(comps.keys())
    data = np.column_stack([comps[k] for k in keys])
    header = ",".join(keys)
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")


def plot_components(out_path: str, comps):
    plt.figure(figsize=(10, 5))
    for k, v in comps.items():
        plt.plot(v, label=k)
    plt.xlabel("step")
    plt.ylabel("reward component")
    plt.title("Reward components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energyplus", default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"))
    parser.add_argument("--model", default="EnergyPlus/5Zone/5ZoneAirCooled.idf")
    parser.add_argument(
        "--weather",
        default="EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    )
    parser.add_argument("--log_dir", default="eplog/reward-components")
    parser.add_argument("--zone_id", default="zone_1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = EnergyPlusMultiAgentEnv(
        energyplus_file=os.path.abspath(args.energyplus),
        model_file=os.path.abspath(args.model),
        weather_file=os.path.abspath(args.weather),
        log_dir=os.path.abspath(args.log_dir),
        seed=args.seed,
        verbose=False,
        framework="ray",
    )
    try:
        comps = run_rollout(env, args.zone_id, args.steps)
    finally:
        env.close()

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"reward_components_{args.zone_id}.csv")
    png_path = os.path.join(args.log_dir, f"reward_components_{args.zone_id}.png")
    save_csv(csv_path, comps)
    plot_components(png_path, comps)
    print(f"[saved] {csv_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
