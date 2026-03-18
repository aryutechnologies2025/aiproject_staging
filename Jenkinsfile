pipeline {
    agent any

    options {
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        IMAGE_NAME = "aiproject-staging"
        IMAGE_TAG = "${BUILD_NUMBER}"
    }

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
                    docker build -t $IMAGE_NAME:$IMAGE_TAG .
                    docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest
                    '''
                }
            }
        }

        stage('Cleanup Old Images (Safe)') {
            steps {
                sh '''
                docker image prune -f

                images=$(docker images $IMAGE_NAME --format "{{.Tag}}" | sort -nr | tail -n +4)

                for tag in $images; do
                    if [ "$tag" != "latest" ]; then
                        docker rmi $IMAGE_NAME:$tag || true
                    fi
                done
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                cd /var/www/ai-fastapi/aiproject_staging

                # Ensure latest container is recreated
                docker-compose down
                docker-compose up -d --force-recreate
                '''
            }
        }
    }
}