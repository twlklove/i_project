#
# Copyright 2021 xx.
#

HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)
CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
CXXFLAGS += -std=c++11
$(info $(SYSTEM))
ifeq ($(SYSTEM), Linux)
LDFLAGS += -L/usr/local/lib `pkg-config --libs protobuf grpc++`\
           -pthread\
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed\
           -ldl
endif
PROTOC = protoc
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_PY_PLUGIN = grpc_python_plugin
grpc_cpp_plugin_path ?= `which $(GRPC_CPP_PLUGIN)`
grpc_py_plugin_path ?= `which $(GRPC_PY_PLUGIN)`

protos_path = src/protos

vpath %.proto $(protos_path)

all:system-check py cc read write server client

src_dir=src
py_src_dir=$(src_dir)/python
cxx_src_dir=$(src_dir)/cxx
vpath %.cpp $(cxx_src_dir)

out_dir=out
cc_out_dir=$(out_dir)/cc
release_dir=$(out_dir)/release
py_release_dir=$(release_dir)/python
$(shell mkdir -p $(out_dir) $(cc_out_dir) $(release_dir) $(py_release_dir))
$(shell export PKG_CONFIG_PATH=$(PKG_CONFIG_PATH):/root/.local/lib/pkgconfig)

cflags=-I $(cxx_src_dir) -I $(cc_out_dir) `pkg-config --cflags protobuf grpc++`   

client: client.cpp hello.pb.cc hello.grpc.pb.cc 
	$(CXX) $(cflags) $(cc_out_dir)/hello.pb.cc $(cc_out_dir)/hello.grpc.pb.cc $< -o $(release_dir)/$@ $(LDFLAGS) `pkg-config --libs protobuf grpc++` 

server: server.cpp hello.pb.cc hello.grpc.pb.cc 
	$(CXX) $(cflags) $(cc_out_dir)/hello.pb.cc $(cc_out_dir)/hello.grpc.pb.cc $< -o $(release_dir)/$@ $(LDFLAGS) `pkg-config --libs protobuf grpc++` 
write:  write.cpp hello.pb.cc 
	$(CXX) $(cflags) $(cc_out_dir)/hello.pb.cc $< -o $(release_dir)/$@ `pkg-config --libs protobuf grpc++`  
read:   read.cpp hello.pb.cc
	$(CXX) $(cflags) $(cc_out_dir)/hello.pb.cc $< -o $(release_dir)/$@ `pkg-config --libs protobuf grpc++`  

cc : hello.pb.cc hello.grpc.pb.cc
py : hello.pb2.py hello.grpc.pb2.py 

.PRECIOUS: %.pb.cc
%.pb.cc: %.proto
	$(PROTOC) -I $(protos_path) --cpp_out=$(cc_out_dir) $<

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(protos_path) --grpc_out=$(cc_out_dir) --plugin=protoc-gen-grpc=$(grpc_cpp_plugin_path) $<

.PRECIOUS: %.pb2.py
%.pb2.py: %.proto
	$(PROTOC) -I $(protos_path) --python_out=$(py_release_dir)  $<
	cp $(py_src_dir)/*.py $(py_release_dir) 

.PRECIOUS: %.grpc.pb2.py
%.grpc.pb2.py: %.proto
	$(PROTOC) -I $(protos_path) --grpc_out=$(py_release_dir) --plugin=protoc-gen-grpc=$(grpc_py_plugin_path) $<

.PHONY:clean
clean:
	rm -rf $(out_dir) *~

# The following is to test your system and ensure a smoother experience.
PROTOC_CMD = which $(PROTOC)
PROTOC_CHECK_CMD = $(PROTOC) --version | grep -q libprotoc.3
PLUGIN_CHECK_CMD = which $(GRPC_CPP_PLUGIN)
HAS_PROTOC = $(shell $(PROTOC_CMD) > /dev/null && echo true || echo false)
ifeq ($(HAS_PROTOC),true)
HAS_VALID_PROTOC = $(shell $(PROTOC_CHECK_CMD) 2> /dev/null && echo true || echo false)
endif
HAS_PLUGIN = $(shell $(PLUGIN_CHECK_CMD) > /dev/null && echo true || echo false)

SYSTEM_OK = false
ifeq ($(HAS_VALID_PROTOC),true)
ifeq ($(HAS_PLUGIN),true)
SYSTEM_OK = true
endif
endif

system-check:
ifneq ($(HAS_VALID_PROTOC),true)
	@echo " DEPENDENCY ERROR"
	@echo "You don't have protoc 3.0.0 installed in your path."
	@echo "Please install Google protocol buffers 3.0.0 and its compiler."
	@echo "You can find it here:"
	@echo "   https://github.com/protocolbuffers/protobuf/releases/tag/v3.0.0"
	@echo
endif
ifneq ($(HAS_PLUGIN),true)
	@echo " DEPENDENCY ERROR"
	@echo "You don't have the grpc c++ protobuf plugin installed in your path."
	@echo "Please install grpc. You can find it here:"
	@echo "   https://github.com/grpc/grpc"
	@echo
endif
ifneq ($(SYSTEM_OK),true)
	@false
endif
