while :; do git push origin main; if [ $? -eq 0 ]; then break; fi; sleep 10; done
