#!/bin/bash

# Check primary Jenkins health
PRIMARY_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://jenkins-primary:8080/health)

if [ "$PRIMARY_HEALTH" != "200" ]; then
    echo "Primary Jenkins unhealthy, initiating failover..."
    
    # Switch DNS to secondary
    aws route53 change-resource-record-sets \
        --hosted-zone-id YOUR_HOSTED_ZONE \
        --change-batch '{
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "jenkins.yourdomain.com",
                    "Type": "A",
                    "TTL": 60,
                    "ResourceRecords": [{"Value":"SECONDARY_JENKINS_IP"}]
                }
            }]
        }'
    
    # Notify team
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"Jenkins Failover Initiated"}' \
        https://hooks.slack.com/services/YOUR_SLACK_WEBHOOK
fi 