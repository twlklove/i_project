kernel_version=$(shell uname -r)

CFLGAS+=
EXTRA_CFLAGS+=-Wno-error=date-time -Wno-date-time


ifneq ($(KERNELRELEASE),)
    $(info "second build")
	param_objs:=test.o
	obj-m:=test.o
else
    $(info "first build")
	KDIR:=/lib/modules/$(kernel_version)/build
    all:
		$(MAKE) $(CFLAGS) $(EXTRA_CFLAGS) -C $(KDIR) M=$(PWD)
.PHONY:clean
clean:
	-rm -rf *.ko modules* Module* *.mod.c *.o *~ *.mod
endif
