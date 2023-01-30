src_files=test_bsearch.c test_qsort.c
obj_files := $(src_files:test_%.c=run_%)
#dst_files := $(src_files:%.c=%.o)

CFLAGS = -g -O2 
LDFLAGS = 
cmd = gcc

all: clean run
	@echo "SRC = $(src_files)"
	@echo "OBJ = $(obj_files)"

run:$(obj_files)
run_%:test_%.c
	$(cmd) $^ $(CFLAGS) $(LDFLAGS) -o $@
	
.PHONY:clean
clean:
	rm -rf $(run_xx) *.o *~
