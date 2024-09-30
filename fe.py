import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random

class EntityValueModel(nn.Module):
    num_units: int  # Number of unique units

    @nn.compact
    def call(self, x):
        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Shared dense layer
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # Output layers
        value_output = nn.Dense(features=1)(x)  # Regression output
        unit_logits = nn.Dense(features=self.num_units)(x)  # Classification output

        return value_output.squeeze(), unit_logits  # Remove last dimension from value_output
