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
                
                docker-compose up -d --build
                '''
            }
        }
    }
}
