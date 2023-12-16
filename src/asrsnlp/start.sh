#!/bin/bash

# project specific setup
setup() {
    local basepath=$1
    local project=$2
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
    setup $basepath $project
else
    echo "No src folder found. Maybe you are not at the right place."
fi
