#  Copyright (c) 2023-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Callable, Dict

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self,batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 3)
        self.agent_radius = kwargs.pop("agent_radius", 0.025)
        self.xdim = kwargs.pop("xdim", 1)
        self.ydim = kwargs.pop("ydim", 1)
        self.spawn_same_pos = kwargs.pop("spawn_same_pos", False)

        self.agent_xspawn_range = 0 if self.spawn_same_pos else self.xdim
        self.agent_yspawn_range = 0 if self.spawn_same_pos else self.ydim
        self.x_semidim = self.xdim - self.agent_radius
        self.y_semidim = self.ydim - self.agent_radius

        self.P = kwargs.pop("P", lambda x: torch.ones(*x.shape[:-1], 1))

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                collide=True,
                shape=Sphere(radius=self.agent_radius),
            )

            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):

        for agent in self.world.agents:
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_xspawn_range, self.agent_xspawn_range),
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_yspawn_range, self.agent_yspawn_range),
                    ],
                    dim=-1,
                ),
                batch_index=env_index,
            )

    def sample_single_env(
        self,
        pos,
    ):
        pos = torch.tensor(pos).to(self.world.device)
        return self.P(pos)

    def sample(
        self,
        pos,
    ):
        pass


    def reward(self, agent: Agent) -> Tensor:
        return torch.zeros(self.world.batch_dim, 1)

    def observation(self, agent: Agent) -> Tensor:
        return torch.zeros(self.world.batch_dim,1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {}

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        from vmas.simulator.rendering import render_function_util

        geoms = [
            render_function_util(
                f=lambda pos: self.sample_single_env(pos, env_index=0),
                plot_range=(self.xdim, self.ydim),
                cmap_alpha=0.5,
            )
        ]

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                * ((self.ydim if i % 2 == 0 else self.xdim) - self.agent_radius)
                + self.agent_radius * 2
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.x_semidim + self.agent_radius
                        if i == 0
                        else -self.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.y_semidim + self.agent_radius
                        if i == 1
                        else -self.y_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
