#!/bin/bash

# Disaster Recovery Script
BACKUP_BUCKET="s3://your-backup-bucket/jenkins-backups/"
JENKINS_HOME="/var/jenkins_home"
LATEST_BACKUP=$(aws s3 ls ${BACKUP_BUCKET} | sort | tail -n 1 | awk '{print $4}')

# Stop Jenkins
systemctl stop jenkins

# Clean current Jenkins home
rm -rf ${JENKINS_HOME}/*

# Restore from latest backup
aws s3 cp ${BACKUP_BUCKET}${LATEST_BACKUP} /tmp/
tar -xzf /tmp/${LATEST_BACKUP} -C /

# Start Jenkins
systemctl start jenkins 