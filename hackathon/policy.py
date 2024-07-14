import torch
from update_prob import update_prob

class Policy:

    def __init__(self, env, world, device):
        super().__init__()
        self.env = env
        self.world = world
        self.device = device

    def run(self, agent):
        action = torch.randn(self.world.batch_dim, self.world.dim_p, device=self.device)
        action = torch.clamp(action, min=-agent.action.u_range, max=agent.action.u_range)
        return action