all:
	gcc -fPIC -shared calculate.c -o libcaculate.so
	gcc -rdynamic -o main main.c -ldl

.PHONY:clean
clean:
	rm -rf *.so main

