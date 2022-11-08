target=run

all:$(target)
$(target):
	g++ test_packet.cpp -lpthread -o $@

.PHONY:clean
clean:
	rm -rf $(target)
