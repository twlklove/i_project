

CFLAGS := -I include

crc32_files := gen_crc32table.c
crc64_files := gen_crc32table.c

all:do_gen_crc32_tbl do_gen_crc64_tbl
do_gen_crc32_tbl:
	gcc $(CFLAGS)  $(crc32_files) -o $@ 

do_gen_crc64_tbl:
	gcc $(CFLAGS)  $(crc64_files) -o $@ 


.PHONY:clean
clean:
	rm -rf do_gen_crc32_tbl do_gen_crc64_tbl
