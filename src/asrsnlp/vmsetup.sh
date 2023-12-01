#!/bin/bash

# Check if Pyenv is already installed
if ! command -v pyenv &> /dev/null; then
    # Install Pyenv with a specific version
    export PYENV_GIT_TAG=v2.3.28
    curl https://pyenv.run | bash
else
    echo "Pyenv is already installed."
fi

# Install essential build dependencies for Python
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Configure environment variables in the user's bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload bash to apply changes
source $HOME/.bashrc

# Install Python version 3.10.12 using Pyenv
pyenv install 3.10.12

# Create and set up a virtual environment named "asrsnlp" using Python 3.10.12
pyenv virtualenv 3.10.12 asrsnlp
pyenv local asrsnlp

echo "Setup completed. Now you can start working on the 'asrsnlp-safrantech' project."