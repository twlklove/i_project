make clean
make CFLAG=-DDEBUG 
valgrind --log-file=valgrind.log --tool=memcheck --leak-check=full --keep-stacktraces=alloc -s ./run
