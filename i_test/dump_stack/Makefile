
CFLAGS=-Wl,-Map=test.map -g -DDEBUG -DCONFIG_STACK_DEPTH=10

all: clean run

run:
	gcc *.c  $(CFLAGS) -rdynamic -o run

.PHONY:clean
clean:
	rm -rf run *~ *.map
