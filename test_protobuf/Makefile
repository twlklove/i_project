all:pb first_read first_write

pb:
	protoc -I=./ --cpp_out=./ *.proto

first_write:
	g++  first_write.cpp *.pb.cc -lprotobuf -o first_write

first_read:
	g++  first_read.cpp *.pb.cc -lprotobuf -o first_read

.PHONY:clean
clean:
	rm -rf first_read first_write log *~
