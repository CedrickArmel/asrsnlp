#!/bin/bash

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

    echo "Setup completed. Now you can start working on the $project project. $python"
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
