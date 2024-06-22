from mmengine import Registry

from .build_env import build_env

AGENTS = Registry("agents")
ENVIRONMENTS = Registry("environments", build_func=build_env)
HOOKS = Registry("hooks")
LOSSES = Registry("losses")
QNETS = Registry("qnets")
REPLAY_BUFFERS = Registry("replay_buffers")
EXPLORERS = Registry("explorers")
