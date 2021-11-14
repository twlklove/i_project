
CROSS_COMPILE :=
ifeq ($(arch), aarch64)
CROSS_COMPILE := aarch64-linux-gnu-
endif

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
LD = $(CROSS_COMPILE)ld
AR = $(CROSS_COMPILE)ar
NM = $(CROSS_COMPILE)nm
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
READELF = $(CROSS_COMPILE)readelf
OBJSIZE = $(CROSS_COMPILE)size
STRIP = $(CROSS_COMPILE)strip 

all:
	$(CC) test.c -o run

.PHONY:clean
clean:
	@rm -rf run *~
