from utils.test_utils import run_episode_for_gradient
import gym
from nn_models.policy import RecA2C

latent_dim = 64
quantization_levels = 7
sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)

name = 'sensor_level_C_a2c_0.07_train.pt'
env = gym.make('CartPole-v1',render_mode='rgb_array')
