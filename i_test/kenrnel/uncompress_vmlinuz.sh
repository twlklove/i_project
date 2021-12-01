/usr/src/linux-headers-`uname -r | cut -d'-' -f1-2`/scripts/extract-vmlinux vmlinuz > vmlinux
readelf -lS vmlinux

#vim ../../arm/qume_install/linux-5.12.4/Documentation/arm64/booting.rst
#vim ../../arm/qume_install/linux-5.12.4/Documentation/filesystems/ramfs-rootfs-initramfs.rst
#Linux kernel 2.6以来，initrd被initramfs(Initial Ram File System)取代。initramfs是rootfs(ramfs的一种)类型的文件，无需挂载，由cpio解压后能够直接被加载到内存。但为了保持一致性，initramfs依然按传统以initrd为名。挂载initramfs后，执行/init脚本，挂载根设备到$rootmnt(默认为/root)下，然后切换根目录，运行真正的init程序

#binwalk vmlinuz | head
#od -A d -t x1  vmlinuz |grep "1f 8b 08 00"
#dd if=vmliuz bs=1 skip=xx | zcat > vmlinux


#lsinitramfs /boot/initrd.img-5.11.0-40-generic
#file /boot/initrd.img-5.11.0-41-generic
#binwalk -y gzip /boot/initrd.img-5.11.0-40-generic
#dd if=/boot/initrd.img-5.11.0-40-generic bs=5693090  skip=1 | zcat | cpio -id --no-absolute-filenames -v
