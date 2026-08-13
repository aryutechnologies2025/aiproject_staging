pipeline {
    agent any

    environment {
        IMAGE = "aryutechnologies2025/aryu_api:latest"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                cd /mnt/storage/projects/ai-fastapi/aiproject_staging

                git pull origin main

                # Pull latest image from Docker Hub
                docker pull $IMAGE

                # Restart container
                docker-compose down
                docker-compose up -d

                # Clean only dangling unused images (DO NOT prune builder cache)
                docker image prune -f
                '''
            }
        }
    }
}
