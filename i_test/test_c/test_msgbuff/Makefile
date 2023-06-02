
src_files:=main.c dev_core.c workqueue.c
CFLAGS:= -g -O2 -D_GNU_SOURCE

target:=run
all:clean $(target) test

$(target):
	gcc $(src_files) -I . $(CFLAGS) -pthread -lpthread -o $@

test:
	gcc test_coroutine.c -o test

.PHONY:clean
clean:
	rm -rf *~  $(target) test
