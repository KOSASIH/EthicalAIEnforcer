import numpy as np
from sklearn.mixture import GaussianMixture

def generate_synthetic_data(original_data, num_samples):
    # Fit a Gaussian Mixture Model on the original data
    gmm = GaussianMixture(n_components=10)
    gmm.fit(original_data)

    # Generate synthetic data samples
    synthetic_data = gmm.sample(num_samples)[0]

    return synthetic_data

# Usage example
original_data = np.array([[1, 2], [3, 4], [5, 6]])  # Replace with your original dataset
num_samples = 100  # Number of synthetic data samples to generate

synthetic_data = generate_synthetic_data(original_data, num_samples)

print(synthetic_data)
