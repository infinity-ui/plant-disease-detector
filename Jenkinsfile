pipeline {
    agent any

    environment {
        IMAGE_NAME = "plant-disea"
        CONTAINER_NAME = "plant-app"
    }

    stages {
        stage('Clone Repo') {
            steps {
                git 'https://github.com/infinity-ui/plant-disease-detector'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Stop Old Container') {
            steps {
                sh 'docker rm -f $CONTAINER_NAME || true'
            }
        }

        stage('Run New Container') {
            steps {
                sh 'docker run -d --name $CONTAINER_NAME -p 8501:8501 $IMAGE_NAME'
            }
        }
    }
}

