import numpy as np


class RandomProjectionEncoder(object):
    def __init__(self, input_shape, latent_dim=3, precision=6):
        self.input_dim = np.prod(input_shape)
        self.proj = np.random.normal(loc=0, scale=1.0 / np.sqrt(latent_dim), size=(latent_dim, self.input_dim))
        self.precision = precision

    def __call__(self, x):
        return np.around(np.dot(self.proj, x.flatten()), self.precision)


def obs_encoder(encoder: RandomProjectionEncoder, obs: np.ndarray) -> tuple:
    encoded_obs = encoder(np.squeeze(obs))
    return tuple(encoded_obs)


def obs_action_encoder(encoder: RandomProjectionEncoder, obs: np.ndarray, action: np.ndarray) -> tuple:
    encoded_obs = encoder(np.squeeze(obs))
    encoded_obs_action = (*tuple(encoded_obs), *tuple(action))
    return encoded_obs_action