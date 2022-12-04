export MALLOC_TRACE=/tmp/mem_leak_data
./run
mtrace ./run $MALLOC_TRACE

