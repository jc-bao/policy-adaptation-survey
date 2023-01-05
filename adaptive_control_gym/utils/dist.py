import torch

def sample_inv_norm(std, size, device='cpu'):
    # sample from truncated normal distribution
    # std recommeded value: 0.4 -> 0.2
    value = torch.zeros(size, device=device)
    torch.nn.init.trunc_normal_(value, mean=0, std=std, a=-1, b=1)
    value = value + 1
    return value - (value>1).type(torch.float32)

if __name__ == '__main__':
    # sample sample_inv_norm and plot the histogram
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set()
    x = sample_inv_norm(0.2, 100000).cpu().numpy()
    plt.hist(x, bins=100)
    # save the plot
    plt.savefig('sample_inv_norm.png')