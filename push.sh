while :; do git push; if [ $? -eq 0 ]; then break; fi; sleep 10; done
