ifeq ($(arch), aarch64)
cross_compile_dir=/usr/local/toolchain/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin
cross_compile_prefix=$(cross_compile_dir)/aarch64-linux-gnu-
else
LD_FLAGS=-Wl,-rpath=/usr/lib/x86_64-linux-gnu/ --static
endif

target=run_test
src_files=*.c
CXX=$(cross_compile_prefix)gcc
CXX_FLAGS=-DTEST -DDEBUG $(debug) -Wl,-Map=outmap.map
#elf default load addr is 0x400000, in /usr/lib/ldscripts/elf_x86_64.x
#LD_FLAGS+=

# -E -S -C
$(target):clean
	${CXX} -I. $(CXX_FLAGS) $(src_files) $(LD_FLAGS) -o $@

.PHONY:clean
clean:
	rm -rf $(target) *~ *.e *.s *.out *.o
