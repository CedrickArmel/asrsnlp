#!/bin/bash

# Install essential build dependencies for Python
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Check if Pyenv is already installed
if ! command -v pyenv &> /dev/null; then
    # Install Pyenv with a specific version
    export PYENV_GIT_TAG=v2.3.28
    curl https://pyenv.run | bash
else
    echo "Pyenv is already installed."
fi

# Configure git
if ! command -v git &> /dev/null; then
    # Install git
    sudo apt update
    sudo apt install -y git
else
    echo "Git is already installed."
fi

# Configure environment variables in the user's bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Configure git
git config --global user.name "CedrickArmel"
git config --global user.email "35418979+CedrickArmel@users.noreply.github.com"
chmod 600 ~/.ssh/id_rsa_vms
# ssh-keygen -t rsa -b 4096 -C "35418979+CedrickArmel@users.noreply.github.com"
# eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_rsa

# Reload bash to apply changes
source $HOME/.bashrc

# project specific setup
setup() {
    local basepath=$1
    local project=$2
    local python=$3

    # Configure environment variables in the user's bashrc
    echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/'"$project"'/conf/local/ml-project-401610-e04a8a987255.json"' >> ~/.bashrc

    # Install Python version
    pyenv install $python

    # Create and set up a virtual environment
    pyenv virtualenv $python $project
    pyenv local $project

    # install requirements in the venv
    pip install -r src/requirements.txt

    # Start mlflow server
    mlflow server --default-artifact-root gs://ml-project-bucket-20231126/$project/mlruns --backend-store-uri $basepath/$project/mlruns --host localhost

    echo "Setup completed. Now you can start working on the '$project' project."
}

# Main script logic

if [ -e ".bashrc" ]; then
    echo "Move to your project's root"
elif [ -d "src" ]; then
    basepath=$(dirname "$PWD")
    project=$(basename "$PWD")
    python=$1
    setup $basepath $project $python
else
    echo "No src folder found. Maybe you are not at the right place."
fi
