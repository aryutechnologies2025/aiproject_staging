pipeline {
    agent any

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

                # Pull latest code
                git pull origin main

                # Build and start container
                docker-compose up -d --build

                # Remove unused images (safe cleanup)
                docker image prune -af

                # Remove build cache (important for your case)
                docker builder prune -af
                '''
            }
        }
    }
}
