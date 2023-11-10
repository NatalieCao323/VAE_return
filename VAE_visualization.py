import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from DataProcessing import inputs  # Make sure this imports your processed data correctly

# Load the trained model
vae = tf.keras.models.load_model("/Users/caoqt1/Desktop/Research/VAE/my_vae_model")
vae.compile(optimizer='adam', loss='some_loss_function') 
def visualize_reconstruction(vae, sample_index):
    # Select a sample from the input data
    original_sample = inputs[sample_index].reshape(1, -1)
    original_sample = tf.cast(original_sample, tf.float32)

    # Use the VAE to reconstruct the sample
    reconstructed_sample = vae(original_sample).numpy()

    # Plot original and reconstructed data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(original_sample[0], label='Original')
    plt.title('Original Data')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(reconstructed_sample[0], label='Reconstructed')
    plt.title('Reconstructed Data')
    plt.legend()
    plt.show()

def generate_data_samples(vae, n_samples, latent_dim):
    # Generate new samples
    random_latent_vectors = tf.random.normal(shape=(n_samples, latent_dim))
    generated_samples = vae.decoder(random_latent_vectors).numpy()

    # Plot generated data
    plt.figure(figsize=(12, 6))
    for i, sample in enumerate(generated_samples):
        plt.plot(sample, label=f'Sample {i+1}')
    plt.title('Generated Data Samples')
    plt.legend()
    plt.show()

# Parameters for visualization
sample_index = 0  # Index of the sample to visualize
n_samples = 10    # Number of new samples to generate
latent_dim = 10   # Dimension of the latent space

# Call the visualization functions
visualize_reconstruction(vae, sample_index)
generate_data_samples(vae, n_samples, latent_dim)
