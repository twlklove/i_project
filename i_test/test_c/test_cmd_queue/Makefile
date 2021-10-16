all:
	gcc --shared -fPIC cmd_queue.c -o libqueue.so
	gcc main.c -L . -lqueue -Wl,-rpath,.  -lpthread -o run

.PHONY:clean
clean:
	@ rm -rf run *~ *.so
