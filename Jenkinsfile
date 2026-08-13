pipeline {
    agent any

    environment {
        IMAGE_NAME = "aryu_api:staging"
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

                # Pull latest changes from git
                git pull origin main

                # Build Docker image locally with BuildKit enabled
                DOCKER_BUILDKIT=1 docker build -t $IMAGE_NAME .

                # Recreate and restart container
                docker compose down
                docker compose up -d --build

                # Clean dangling unused images
                docker image prune -f
                '''
            }
        }
    }
}
