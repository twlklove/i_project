CFLAGS:=$(CFLAG)

ifdef ASAN_CHECK
$(info "asan check")
ASAN_CFG:=-fsanitize=address 
ASAN_CFG+=-fsanitize-recover=address 
ASAN_CFG+=-fno-stack-protector 
ASAN_CFG+=-fno-omit-frame-pointer 
ASAN_CFG+=-fno-var-tracking -g2        #release use -g1
CFLAGS += $(ASAN_CFG)  
endif

project_dir=..
base_dir=$(project_dir)/base
json_dir=$(base_dir)/cJSON
main_dir=$(project_dir)/main

CFLAGS+= -I $(main_dir) -I $(base_dir) -I $(json_dir) -Werror -O2 -g

src_files=$(main_dir)/main.c $(base_dir)/config.c $(base_dir)/log.c $(json_dir)/cJSON.c

output=run
run:
	gcc $(CFLAGS) $(src_files) -o $(output)

.phony:clean
clean:
	rm -rf $(output) *~  *.log  *.log.*
