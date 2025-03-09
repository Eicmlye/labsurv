from mmcv.utils import Registry

from .build_env import build_env

AGENTS = Registry("agents")
ENVIRONMENTS = Registry("environments", build_func=build_env)
EXPLORERS = Registry("explorers")
HOOKS = Registry("hooks")
IMITATORS = Registry("imitators")
LOSSES = Registry("losses")
STRATEGIES = Registry("strategies")
REPLAY_BUFFERS = Registry("replay_buffers")
RUNNERS = Registry("runners")
