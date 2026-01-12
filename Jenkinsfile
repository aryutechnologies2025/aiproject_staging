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
                sh 'docker build -t aiproject-staging:latest .'
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                cd /var/www/ai-fastapi/aiproject_staging
                docker-compose down
                docker-compose up -d
                '''
            }
        }
    }
}

