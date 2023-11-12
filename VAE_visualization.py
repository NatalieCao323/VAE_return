import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from DataProcessing import inputs, samples, targets  # Make sure this imports your processed data correctly
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

def generate_data_samples(vae, n_samples, latent_dim, max_plots=10):
    # Generate new samples
    random_latent_vectors = tf.random.normal(shape=(n_samples, latent_dim))
    generated_samples = vae.decoder(random_latent_vectors).numpy()

    #Plot a subset of the generated data
    plt.figure(figsize=(12, 6))
    num_samples_to_plot = min(max_plots, len(generated_samples))  # Limit the number of plots
    for i in range(num_samples_to_plot):
        plt.plot(generated_samples[i], label=f'Sample {i+1}')
    plt.title('Generated Data Samples')
    plt.legend()
    plt.show()
    return generated_samples

def evaluate_model(vae, inputs, targets, latent_dim):
    # Generate new samples (stock prices)
    generated_samples = generate_data_samples(vae, len(inputs), latent_dim)
    # Rescale the generated and target data if they were scaled
    generated_samples = generated_samples.mean(axis=1) if generated_samples.ndim > 1 else generated_samples

    # Ensure generated_samples and targets have the same shape
    if generated_samples.shape != targets.shape:
        raise ValueError(f"Shape mismatch: targets shape {targets.shape}, generated_samples shape {generated_samples.shape}")

    # Calculate MSE
    mse = mean_squared_error(targets, generated_samples)
    print(f"Mean Squared Error: {mse}")

    # Visualization for the first few samples
    for i in range(min(5, len(inputs))):  # Adjust the range as needed
        plt.figure(figsize=(10, 4))
        plt.plot(targets[i], label='Actual')
        plt.plot(generated_samples[i], label='Generated')
        plt.title(f"Stock Prices - Sample {i}")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


# Call the visualization and evaluation functions
sample_index = 0  # Index of the sample to visualize
n_samples = 10    # Number of new samples to generate
latent_dim = 10   # Dimension of the latent space

visualize_reconstruction(vae, sample_index)
generate_data_samples(vae, n_samples, latent_dim)
evaluate_model(vae, inputs, targets, latent_dim)
