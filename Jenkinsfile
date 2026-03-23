pipeline {
    agent any

    environment {
        IMAGE = "sivaarun10/aryu_api:latest"
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

                # Cleanup
                docker image prune -af
                docker builder prune -af
                '''
            }
        }
    }
}