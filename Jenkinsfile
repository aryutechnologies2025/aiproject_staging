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

                git pull origin main

                # STOP old container safely
                docker-compose down

                # BUILD + START fresh container
                docker-compose up -d --build

                # CLEAN unused images AFTER new container is running
                docker image prune -af
                docker builder prune -af
                '''
            }
        }
    }
}
