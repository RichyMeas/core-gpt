## Install the environment
1. After cloning the github repo, cd into the root directory which is the parent of the src, tests directories
2. pip install uv
3. Run uv init

## Running unit tests 
1. Go to the root directory
2. Run `uv run --python 3.10 --with pytest-cov --with '.[tests]' pytest --cov=src`

If you want HTML report then do the following commands:
2a. Run `uv run --python 3.10 --with pytest-cov --with '.[tests]' pytest --cov=src --cov-report=html`
2b. This will make a htmlconv directory at the root and then run `start htmlcov\index.html` on Windows or `open htmlcov/index.html` on Apple MacOS or `xdg-open htmlcov/index.html` on Linux