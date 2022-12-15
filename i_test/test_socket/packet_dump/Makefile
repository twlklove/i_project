target=packet_dump

CFLAGS += -Wno-address-of-packed-member

all:clean $(target)
$(target):
	g++ -I . $(CFLAGS) test_packet.cpp checksum.c -lpthread -o $@

.PHONY:clean
clean:
	rm -rf $(target) *~
