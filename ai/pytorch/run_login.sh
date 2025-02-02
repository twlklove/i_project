#!/bin/bash
jupyter notebook  --allow-root --ip=192.168.1.5 &

while :; do echo hello>/dev/null; sleep 10; done &
