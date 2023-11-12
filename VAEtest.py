import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from DataProcessing import inputs, scaler, mean_value, std_dev_value
import os

# Define the VAE model
class VAE(Model):
    def __init__(self, latent_dim=2, mean=0.0, std_dev=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mean = mean
        self.std_dev = std_dev
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(60,)),
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(latent_dim + latent_dim, kernel_initializer='he_normal')
            ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.Dropout(0.2),  # Assuming normalized data between 0 and 1
            layers.Dense(60, activation='linear', kernel_initializer='he_normal'),
            ])
    
    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        # Use tf.shape to get the runtime shape
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, apply_sigmoid=True)
    
        # Loss function for VAE
    def custom_loss(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_logit = self.decode(z)

        # MSE for reconstruction loss
        mse_loss = tf.reduce_mean(tf.square(x - x_logit))

        # KL divergence, with numerical stability
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var + 1e-8), axis=1)
        return tf.reduce_mean(mse_loss + kl_loss)

# Instantiate the VAE model with the correct mean and std_dev
mean_value = scaler.mean_[0]
std_dev_value = np.sqrt(scaler.var_[0])
vae = VAE(latent_dim=10, mean=mean_value, std_dev=std_dev_value)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

# Training step
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = model.custom_loss(tf.cast(x, tf.float32))  # Call the custom loss function
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_value(g, -1., 1.) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Training loop
def train_vae(model, data, epochs, patience=10):
    best_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for batch in data:
            loss = train_step(model, batch, optimizer)
            epoch_loss_avg.update_state(loss)
        
        current_loss = epoch_loss_avg.result()
        print('Epoch: {}, Loss: {}'.format(epoch, current_loss))

        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Stopping early due to lack of improvement in loss.")
                break

# Assuming stock_returns is a NumPy array of shape (num_samples, 60)
# Ensure inputs are in float32
inputs = inputs.astype('float32')
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# Explicitly build the model
vae.build(input_shape=(None, 60))
# Create a dummy input batch with the correct shape
# Assuming your input data shape is (60,), create a dummy batch with shape (1, 60)
dummy_input = tf.random.normal([1, 60])

# Call the model on the dummy input to build it
_ = vae(dummy_input)

# Now try saving the model
model_save_path = "/Users/caoqt1/Desktop/Research/VAE/my_vae_model"
vae.save(model_save_path)


# Generate data
def generate_stock_returns(model, n_samples=10):
    # Sample from the latent space
    z = tf.random.normal(shape=(n_samples, model.latent_dim))
    # Decode the latent space samples to stock returns
    generated_returns = model.decode(z, apply_sigmoid=True)
    return generated_returns.numpy()

# Generate new data
new_stock_returns = generate_stock_returns(vae, n_samples=60)
# Post-process generated data if necessary (e.g., un-normalizing)

