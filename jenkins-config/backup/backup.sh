#!/bin/bash

# Backup Jenkins configuration
BACKUP_DIR="/jenkins-backup"
DATE=$(date +%Y%m%d_%H%M%S)
JENKINS_HOME="/var/jenkins_home"

# Create backup directory if it doesn't exist
mkdir -p ${BACKUP_DIR}

# Backup Jenkins home directory
tar -czf ${BACKUP_DIR}/jenkins_home_${DATE}.tar.gz ${JENKINS_HOME}

# Keep only last 7 days of backups
find ${BACKUP_DIR} -type f -name "jenkins_home_*.tar.gz" -mtime +7 -exec rm {} \;

# Sync to secondary region (assuming AWS)
aws s3 sync ${BACKUP_DIR} s3://your-backup-bucket/jenkins-backups/

# Additional backup for monitoring data
MONITORING_BACKUP_DIR="/monitoring-backup"
mkdir -p ${MONITORING_BACKUP_DIR}

# Backup Prometheus data
kubectl cp prometheus-pod:/prometheus-data ${MONITORING_BACKUP_DIR}/prometheus-${DATE}.tar.gz

# Backup Grafana dashboards
kubectl cp grafana-pod:/var/lib/grafana ${MONITORING_BACKUP_DIR}/grafana-${DATE}.tar.gz

# Sync monitoring backups to S3
aws s3 sync ${MONITORING_BACKUP_DIR} s3://your-backup-bucket/monitoring-backups/ 