
src_dir=src
py_src_dir=$(src_dir)/python
c++_src_dir=$(src_dir)/c++

out_dir=out
release_dir=$(out_dir)/release
py_release_dir=$(release_dir)/python
$(shell mkdir -p $(out_dir) $(release_dir) $(py_release_dir))

cflags=-I $(c++_src_dir) -I $(out_dir)

all:cpp py

cpp_pb :
	protoc -I=src --cpp_out=$(out_dir) --proto_path=src src/*.proto	
cpp : cpp_pb
	g++  $(cflags) $(c++_src_dir)/write.cpp $(out_dir)/*.pb.cc -o $(release_dir)/write `pkg-config --cflags --libs protobuf` 
	g++  $(cflags) $(c++_src_dir)/read.cpp $(out_dir)/*.pb.cc -o $(release_dir)/read `pkg-config --cflags --libs protobuf`
py:
	protoc -I=src --python_out=$(py_release_dir) --proto_path=src src/*.proto
	cp $(py_src_dir)/*.py $(py_release_dir) 

.PHONY:clean
clean:
	rm -rf $(out_dir) *~
