import numpy as np
from torch.distributions import MultivariateNormal
import torch


def generate_ground_truth(num_samples, 
                          rej_batch_size = 500000,
                          true_obs = (0.31444563,-1.37717961, 
                                      1.69723032, -0.92678205, 
                                      5.54504172, -1.0198386, 
                                      2.35527439, -0.90912185),
                          eps = 1.5,
                          seed = None):
    """Given an observation X, generate samples from the distribution
    P(\theta | dist(X, true_obs) <= \eps).
    via rejection sampling.
    
    Args:
        num_samples (int): Number of samples to generate."""
    
    # generate ground truth via rejection sampling

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_transformed_gaussian(num_samples):
        """Generate data from a transformed Gaussian distribution.
        
        Args:
            num_samples (int): Number of samples to generate."""
        
        thetas = np.random.uniform(-3, 3, 5 * num_samples)
        thetas = thetas.reshape(num_samples, 5)
        fake_mu = thetas[:, :2]
        fake_s1 = thetas[:, 2]**2
        fake_s2 = thetas[:, 3]**2
        fake_rho = np.tanh(thetas[:, 4])
        fake_cov = fake_rho * fake_s1 * fake_s2
        fake_Sigma = np.asarray([fake_s1**2, fake_cov, fake_cov, 
                                 fake_s2**2])
        fake_Sigma = fake_Sigma.T.reshape(-1, 2, 2)
        data = MultivariateNormal(torch.tensor(fake_mu), 
                                  torch.tensor(fake_Sigma)).sample([4])
        data = data.permute(1, 0, 2).reshape(num_samples, 4*2)
        data = data.detach().numpy()
        return data, thetas
    
    true_obs = np.asarray(true_obs)
    
    retained_thetas = []
    retained_data = []
    while len(retained_thetas) < num_samples:
        data, thetas = generate_transformed_gaussian(rej_batch_size)
        ret_idx = np.where(np.all(np.abs(data - true_obs) <= eps, axis=1))[0]
        retained_thetas.extend(thetas[ret_idx, :])
        retained_data.extend(data[ret_idx, :])
    
    return np.array(retained_data[:num_samples]), np.array(retained_thetas[:num_samples])