/usr/src/linux-headers-`uname -r | cut -d'-' -f1-2`/scripts/extract-vmlinux vmlinuz > vmlinux
readelf -lS vmlinux

#binwalk vmlinuz | head
#od -A d -t x1  vmlinuz |grep "1f 8b 08 00"
#dd if=vmliuz bs=1 skip=xx | zcat > vmlinux
