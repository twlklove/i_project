

CFLAGS := -I include -I include/byteorder -DCONFIG_CRC32_SLICEBY8

crc32_files := gen_crc32table.c
crc64_files := gen_crc32table.c
crc32_test_file := crc32.c crc32test.c

crc32_tbl := include/crc32table.h

all:clean do_gen_crc32_tbl do_gen_crc64_tbl crc32_tbl crc32_test
	
do_gen_crc32_tbl:
	gcc $(CFLAGS)  $(crc32_files) -o $@ 
	
do_gen_crc64_tbl:
	gcc $(CFLAGS)  $(crc64_files) -o $@ 

crc32_tbl:
	$(shell ./do_gen_crc32_tbl > $(crc32_tbl))

crc32_test:
	gcc $(CFLAGS)  $(crc32_test_file) -o $@

.PHONY:clean
clean:
	rm -rf do_gen_crc32_tbl do_gen_crc64_tbl $(crc32_tbl) crc32_test 
