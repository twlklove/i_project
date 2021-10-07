target := read write
CFLAGS :=
LDFLAGS :=

all:$(target)
read:read.c
	gcc $(CFLGAS) $(LDFLAGS) $^ -o $@

write:write.c
	gcc $(CFLGAS) $(LDFLAGS) $^ -o $@

.PHONY:clean
clean:
	rm -rf $(target)
