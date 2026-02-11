
## 1. Setup

### 1.1 Python Version and Virtual Environment

For this project we use Python version 3.12. Many IBM watsonx libraries require version 3.11 or later. To standardize the Python version across the project we use `pyenv`.
#### 1.1.1 Linux (Ubuntu)

Install build dependencies
```bash
sudo apt update
sudo apt install -y build-essential curl git \
libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
libffi-dev liblzma-dev python3-openssl
```

Install pyenv

```bash
curl https://pyenv.run | bash
```

Add pyenv to the shell
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
```

install the version
```bash
pyenv install 3.12.3
```

set the python version in your project
```bash
cd myproject #substitute myproject with actual folder name
pyenv local 3.12.3 # creates a .python-version file
python --version # check --> should display Python 3.12.3
```

create the virtual environment in the project, activate it and install necessary dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you install additional dependencies, always save them to the `requirements.txt` before you commit to the repo:

```bash
pip freeze > requirements.txt
```

#### 1.1.2 macOS

Install homebrew (if you don't already have it)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add brew to the shell
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Install pyenv and necessary dependencies
```bash
brew install pyenv
brew install openssl readline sqlite3 xz zlib tcl-tk
```

add pyenv to the shell
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
exec "$SHELL"
```

install the version
```bash
pyenv install 3.12.3
```

set the python version in your project
```bash
cd myproject #substitute myproject with actual folder name
pyenv local 3.12.3 # creates a .python-version file
python --version # check --> should display Python 3.12.3
```

create the virtual environment in the project, activate it and install necessary dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you install additional dependencies, always save them to the `requirements.txt` before you commit to the repo:

```bash
pip freeze > requirements.txt
```


