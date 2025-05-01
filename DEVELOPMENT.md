## Install the environment
1. After cloning the github repo, cd into the root directory which is the parent of the src, tests directories
2. `pip install uv`
3. Run `uv init`

## Running unit tests 
1. Go to the root directory
2. Run `uv run --python 3.10 --with pytest-cov --with '.[tests]' pytest --cov=src`

If you want HTML report then do the following commands:
2a. Run `uv run --python 3.10 --with pytest-cov --with '.[tests]' pytest --cov=src --cov-report=html`
2b. This will make a htmlconv directory at the root and then run `start htmlcov\index.html` on Windows or `open htmlcov/index.html` on Apple MacOS or `xdg-open htmlcov/index.html` on Linux

If you want to run 1 unit test you can run:
1. `uv run --python 3.10 --with pytest-cov --with '.[tests]' pytest --cov=src -v path/to/your/test_file.py`
The `-v` will show all specific unit test functions that passed or failed.

## Run scripts
1. Go to root directory
2. Run `uv run --python 3.10 -m src.scripts.paramater_count_calc --config-name model.yaml --config-path \path\to\parent\of\model.yaml`

## Setting up CUDA and PyTorch
1. Install WSL for Windows users (may need to restart your computer after installation)
1a. Open Windows PowerShell as Administrator
1b. Run `wsl --install`
2. Install the Remote WSL extension on VSCode by running the below in command prompt:
2a. `code --install-extension ms-vscode-remote.remote-wsl`
3. Open WSL inside VSCode and verify you are on a Linux system
3a. In the search bar type: `>WSL: Remote Open in WSL`
3b. Press Control + ` (I think to open terminal)
3a. Run `uname -a` and you should see something like `Linux <hostname> 5.10.102.1-microsoft-standard-WSL2 #1 SMP ... x86_64 GNU/Linux`
4. Install pip in the WSL environment by running the following [Note: the environment should default come with Python 3.12.2 with this]:
4a. `sudo apt update`
4b. `sudo apt install python3-pip -y`
5. Install venv and activate the venv by running the following:
5a. `sudo apt install python3-venv -y`
5b. `python3 -m venv ~/core-gpt-venv`
5c. `source ~/core-gpt-venv/bin/activate`
6. Export your PYTHONPATH as follows but `export PYTHONPATH="/path/to/core-gpt/src"`
7. `pip3 install triton`
8. For Linux CUDA 12.6 users you can download PyTorch by running `pip3 install torch torchvision torchaudio` (other commands for different systems
can be found here https://pytorch.org/get-started/locally/)
9. Install dependencies for the project `pip3 install datasets==3.5.0 "hydra-core>=1.3.2" numpy==2.2.4 "omegaconf>=2.3.0" "pytest>=8.3.5" "pytest-cov>=6.1.1" regex==2024.11.6 requests==2.32.3 tiktoken==0.9.0 tqdm==4.66.5 tensorboard`
10. Install cuda toolkit `sudo apt install nvidia-cuda-toolkit`

To run pytests in this setup please run the following:
1. `pytest --cov=src`

To train a model in this setup please run the following:
1. `python -m src.entrypoints.train_gpt --config-name model.yaml --config-path /path/to/parent/of/model_yaml`

To use parameter count scripts:
1. `python -m src.scripts.parameter_count_calc --config-name model.yaml --config-path /path/to/core-gpt/experiments/3m_baseline/`