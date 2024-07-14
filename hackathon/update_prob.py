import torch

def update_prob(x, P):
    '''
    x is of dim batch_size x # points x 2
    '''
    return 1 - (1 - P(x[:,:,0]))*(1-P(x[:,:,1]))

# Tests
# x1 = torch.arange(1, 6, 1)
# x2 = torch.arange(2, 7, 1)
# test_x = torch.unsqueeze(torch.stack((x1, x2), 1), 2)
# test_x = test_x.float()
# test_x = test_x.view(test_x.shape[-1], test_x.shape[1], -1)

# new = update_prob(test_x, 1)
# print(test_x, new)
