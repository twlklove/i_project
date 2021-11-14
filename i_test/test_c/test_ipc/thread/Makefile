
targets:=automic 

all:$(targets)

automic:
	 g++ automic.cpp -std=c++11  -lpthread -o $@

.PHONY:clean
clean:
	rm -rf $(targets)
