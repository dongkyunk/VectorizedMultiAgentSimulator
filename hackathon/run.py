import time

import torch

from vmas import make_env
from vmas.simulator.utils import save_video
from policy import Policy
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily


def run(
    n_steps: int = 200,
    n_envs: int = 32,
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
):

    env_kwargs = {
        "P": EnvP(n_envs)
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


class EnvP:

    def __init__(self, batch_dim):
        self.batch_dim = batch_dim
        self.n_modes = 6
        self.dist = None
        self.reset()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset(self):
        loc = torch.rand(self.batch_dim, self.n_modes, 2) * 2 - 1
        std = torch.exp(torch.randn(self.batch_dim, self.n_modes, 2) / 4 - 0.6)
        comp = Independent(Normal(loc=loc, scale=std), 1)
        weights = Categorical(torch.ones(self.batch_dim, self.n_modes))
        gmm = MixtureSameFamily(weights, comp)
        self.dist = gmm

    def forward(self, x, env_index=None):
        if env_index is None:
            return torch.exp(self.dist.log_prob(x))
        else:
            comp_dist = self.dist._component_distribution
            mix_dist = self.dist._mixture_distribution
            loc = comp_dist.base_dist.loc[env_index]
            std = comp_dist.base_dist.scale[env_index]
            weights = mix_dist.probs[env_index]
            comp_new = Independent(Normal(loc=loc, scale=std), 1)
            mix_new = Categorical(weights)
            dist = MixtureSameFamily(mix_new, comp_new)
            return torch.exp(dist.log_prob(x))


if __name__ == "__main__":
    run(
        n_envs=2,
        n_steps=500,
        render=True,
        save_render=False,
    )
