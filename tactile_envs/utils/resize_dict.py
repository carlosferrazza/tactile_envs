"""Wrapper for resizing observations."""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box, Dict


class ResizeDict(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Resize the image observation.

    This wrapper works on environments with image observations. More generally,
    the input can either be two-dimensional (AxB, e.g. grayscale images) or
    three-dimensional (AxBxC, e.g. color images). This resizes the observation
    to the shape given by the 2-tuple :attr:`shape`.
    The argument :attr:`shape` may also be an integer, in which case, the
    observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env, shape: tuple[int, int] | int, pixel_key='pixels') -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

        self.pixel_key = pixel_key

        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

        self.shape = tuple(shape)

        assert isinstance(
            env.observation_space[self.pixel_key], Box
        ), f"Expected the observation space to be Box, actual type: {type(env.observation_space)}"
        dims = len(env.observation_space[self.pixel_key].shape)
        assert (
            dims == 2 or dims == 3
        ), f"Expected the observation space to have 2 or 3 dimensions, got: {dims}"

        obs_shape = self.shape + env.observation_space[self.pixel_key].shape[2:]
        self.observation_space = Dict({self.pixel_key: Box(low=0, high=1, shape=obs_shape, dtype=np.float64)})

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        observation[self.pixel_key] = cv2.resize(
            observation[self.pixel_key], self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        observation[self.pixel_key] = observation[self.pixel_key].reshape(self.observation_space[self.pixel_key].shape[-3:])/255
        return observation
