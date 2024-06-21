[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# labsurv

This repository is a reinforcement learning solution to the Optimal Camera Placement (OCP) problem.

## Installation

```bash
# build virtual environment
python -m venv .

# install required packages
pip install -r requirements.txt
pip install -e .
```

## Code Style

This repository uses `black`, `isort` and `flake8` for code formatting. To format changed files, run

```bash
sh infra/scripts/format.sh --git
```

And find more formatting options with `--help`.
