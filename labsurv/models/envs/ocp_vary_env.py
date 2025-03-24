import os
import os.path as osp
from typing import Dict, List, Optional

import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv, BaseSurveillanceEnv


@ENVIRONMENTS.register_module()
class OCPVaryEnv(BaseEnv):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        member_cfgs: List[Dict],
        save_path: str,
    ):
        """
        ## Description:

            This environment collects some MAPPO surveillance room environment class
            and randomly gives out one of them.
        """
        super().__init__()

        self._members: List[BaseSurveillanceEnv] = []
        for cfg in member_cfgs:
            self._members.append(ENVIRONMENTS.build(cfg))

        self.working_env_index: Optional[int] = None
        self.env_order_cache: List[int] = []

    def __len__(self):
        return len(self._members)

    def _assert_working(self):
        assert (
            self.working_env_index is not None
        ), "Working env not activated yet. Call `reset()` first."

    @property
    def working_env(self):
        self._assert_working()

        return self._members[self.working_env_index]

    @property
    def info_room(self):
        self._assert_working()

        return self.working_env.info_room

    @property
    def agent_num(self):
        self._assert_working()

        return self.working_env.agent_num

    def step(self, **kwargs):
        return self.working_env.step(**kwargs)

    def reset(self, seed: Optional[int] = None):
        # do env init works
        super().reset(seed=seed)

        self.working_env_index = self._np_random.choice(len(self))
        self.env_order_cache.append(self.working_env_index)

        return self.working_env.reset()

    def save(self, save_path: str, **kwargs):
        kwargs["save_path"] = save_path

        if save_path.endswith(".txt"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            env_order_save_path = ".".join(save_path.split(".")[:-1]) + ".txt"
        else:
            os.makedirs(save_path, exist_ok=True)
            env_order_save_path = osp.join(save_path, "env_order.txt")

        with open(env_order_save_path, "a+") as f:
            for index in self.env_order_cache:
                f.write(index)
                f.write("\n")

        return self.working_env.save(**kwargs)
