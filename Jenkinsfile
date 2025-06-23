pipeline {
    agent any

    environment {
        IMAGE_NAME = "plant-disea"
        CONTAINER_NAME = "plant-app"
    }

    stages {
        stage('Clone Repo') {
            steps {
                git ''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Stop Old Container (if any)') {
            steps {
                sh 'docker rm -f $CONTAINER_NAME || true'
            }
        }

        stage('Run Streamlit App in Docker') {
            steps {
                sh 'docker run -d --name $CONTAINER_NAME -p 8501:8501 $IMAGE_NAME'
            }
        }
    }
}
