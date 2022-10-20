import gym
import pytest

import cfrl


@pytest.mark.parametrize("num_envs", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("env_id", ["CartPole-v0", "ALE/SpaceInvaders-v5"])
@pytest.mark.parametrize("seeds", [0, 1, 2])

class TestVectorEnv:
    @pytest.fixture(autouse=True)
    def SetUp(self, num_envs, env_id, seeds):
        self.num_envs = num_envs
        self.env_id = env_id
        self.seeds = seeds


        