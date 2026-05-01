#!/bin/bash

ADDITIONAL_COMMENT="$1"
git add .
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
if [ -n "$ADDITIONAL_COMMENT" ]; then
    git commit -m "$TIMESTAMP: $ADDITIONAL_COMMENT"
else
    git commit -m "$TIMESTAMP" -m "Updated"
fi
git push origin HEAD
