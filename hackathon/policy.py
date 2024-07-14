import torch

class Policy:

    def __init__(self, env, world, device):
        super().__init__()
        self.env = env
        self.world = world
        self.device = device
        self.EnvP = self.env.scenario.EnvP

    def run(self, agent):
        pos = agent.state.pos
        pos.requires_grad_(True)
        V = self.EnvP(pos)
        dVdx = torch.autograd.grad(V.sum(), pos)[0]
        dVdx_mag = dVdx.norm(dim=-1, keepdim=True)
        dVdx_dir = dVdx / dVdx_mag
        return dVdx_dir