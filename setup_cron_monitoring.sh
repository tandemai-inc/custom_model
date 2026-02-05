#!/bin/bash
# Setup cron job to monitor training progress
# Run this script to add a cron job that checks every 5 minutes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_permeability.py"
LOG_FILE="$SCRIPT_DIR/training_monitor.log"

echo "Setting up cron monitoring for training progress..."
echo ""
echo "This will add a cron job that runs every 5 minutes"
echo "Monitor script: $MONITOR_SCRIPT"
echo "Log file: $LOG_FILE"
echo ""

# Create cron entry (runs every 5 minutes)
CRON_ENTRY="*/5 * * * * cd $SCRIPT_DIR && conda run -n xgboost_training python $MONITOR_SCRIPT >> $LOG_FILE 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$MONITOR_SCRIPT"; then
    echo "⚠️  Cron job already exists!"
    echo "Current cron jobs:"
    crontab -l | grep "$MONITOR_SCRIPT"
    echo ""
    read -p "Do you want to remove the existing job and add a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing entry
        crontab -l 2>/dev/null | grep -v "$MONITOR_SCRIPT" | crontab -
        echo "Removed existing cron job"
    else
        echo "Keeping existing cron job"
        exit 0
    fi
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✓ Cron job added successfully!"
echo ""
echo "The monitor will run every 5 minutes and log to: $LOG_FILE"
echo ""
echo "To view the log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To remove the cron job later:"
echo "  crontab -e"
echo "  (then delete the line with monitor_permeability.py)"
echo ""
echo "To check current cron jobs:"
echo "  crontab -l"

