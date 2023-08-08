
CFLAGS=-Wl,-Map=test.map -g -DCONFIG_STACK_DEPTH=10 -rdynamic 

ifeq ($(test), 1)    #make test=1
CFLAGS+=-DDEBUG -g3
endif

all: clean run

run:
	gcc *.c  $(CFLAGS)  -o run

.PHONY:clean
clean:
	rm -rf run *~ *.map
