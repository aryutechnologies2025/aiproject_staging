pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                retry(2) {
                    sh '''
                    docker build --no-cache -t aryu_api:latest .
                    '''
                }
            }
        }

        stage('Cleanup Old Images') {
            steps {
                sh '''
                docker image prune -af || true
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                cd /var/www/ai-fastapi/aiproject_staging
                docker-compose down
                docker-compose up -d --build
                '''
            }
        }
    }
}