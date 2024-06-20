from copy import deepcopy

import gym
from labsurv.utils import WARN
from mmengine import ConfigDict, Registry, build_from_cfg


def build_env(env_cfg: ConfigDict, registry: Registry):
    cfg = deepcopy(env_cfg)
    if "id" in cfg.keys():
        id = cfg.pop("id")
        if "type" in cfg.keys():
            cfg.pop("type")
            env_cfg.pop("type")
            print(WARN(f"Both `id` and `type` found in config, use gym env `{id}`."))
        return gym.make(id, **cfg)
    elif "type" in cfg.keys():
        return build_from_cfg(cfg, registry)

    raise KeyError("Neither `id` nor `type` is found in the config file.")
