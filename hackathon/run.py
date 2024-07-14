import time

import torch

from vmas import make_env
from vmas.simulator.utils import save_video
from policy import Policy


def run(
    n_steps: int = 200,
    n_envs: int = 32,
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
):

    P = lambda x: torch.ones(*x.shape[:-1], 1)

    env_kwargs = {
        "P": P
    }

    scenario_name = "search"

    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    policy = Policy(env=env, world=env.world, device=device)

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    for _ in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i, agent in enumerate(env.agents):
            actions[i] = policy.run(agent)
        obs, rews, dones, info = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )


if __name__ == "__main__":
    run(
        n_envs=2,
        n_steps=200,
        render=True,
        save_render=False,
    )
