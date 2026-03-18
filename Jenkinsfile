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

        stage('Build & Deploy') {
 
            steps {
                sh '''
                cd /mnt/storage/projects/ai-fastapi/aiproject_staging
                
                git pull origin main
                
                docker-compose down
                docker-compose up -d --build
                '''
            }
        }
    }
}
