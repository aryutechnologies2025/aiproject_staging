pipeline {
    agent any

    options {
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        IMAGE_NAME = "aryu_api"
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
                # Remove dangling images only (safe)
                docker image prune -f

                # Keep last 3 versions, delete older ones
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

                # Pull latest image tag (optional if using registry)
                # docker pull $IMAGE_NAME:latest

                docker-compose down
                docker-compose up -d
                '''
            }
        }
    }
}