from jsonargparse import ArgumentParser
from openrl.configs.utils import ProcessYamlAction
from openrl.configs.config import create_config_parser

parser = create_config_parser()

args = parser.parse_args()

import pdb; pdb.set_trace()  # EM