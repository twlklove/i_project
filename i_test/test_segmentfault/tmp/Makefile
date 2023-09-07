ifeq ($(arch), aarch64)
	cross_compile_dir=/opt/....usr/bin/
	cross_compile_prefix=$(cross_compile_dir)/aarch64-linux-gnu-
else
	LD_FLAGS=#-Wl,-rpath=/usr/lib/x86_64-linux-gnu/ --static
endif

target=i_trace
src_files=$(CURDIR)/*.c
CXX=$(cross_compile_prefix)gcc
CXX_FLAGS=-DTEST -DDEBUG
LD_FLAGS+=

run:clean $(target)

$(target):
	$(CXX) -I . $(CXX_FLAGS) $(src_files) $(LD_FLAGS) -o $@

.PHONY:clean
clean:
	rm -rf $(target) *~ *.o
