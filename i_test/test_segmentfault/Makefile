
CFLAGS=-Wl,-Map=test.map -g -DCONFIG_STACK_DEPTH=10 -rdynamic -I ../dump_stack 

ifeq ($(debug), 1)    #make debug=1
CFLAGS+=-g3
endif

src_files=*.c ../dump_stack/*.c

all: clean run

run:
	gcc $(src_files)  $(CFLAGS)  -o run

.PHONY:clean
clean:
	rm -rf run *~ *.map
