#!/bin/bash
python_script="server.py"
log_file="server.log"

is_process_running() {
    count=$(ps aux | grep -v "grep" | grep "$python_script" | wc -l)
    if [ "$count" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

if is_process_running; then
    echo "Keras Service is already running"
    exit 1
fi

python "$python_script" >> "$log_file" 2>&1 &
echo "Keras Service started, log file: $log_file"
