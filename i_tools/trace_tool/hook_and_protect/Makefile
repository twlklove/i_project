

CFLAGS:= $(debug) -g
run:clean make_so
	gcc test.c $(CFLAGS) -ldl  -o  run

make_so:
	gcc test_lib.c $(CFLAGS) -shared -fpic -o libtest.so

.PHONY:clean
clean:
	rm -rf run *~ *.so
