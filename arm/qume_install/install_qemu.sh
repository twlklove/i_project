install_cmd="apt-get install"
${install_cmd} gcc-aarch64-linux-gnu
${install_cmd}  qemu-system-arm

wget https://busybox.net/downloads/busybox-1.33.1.tar.bz2
cp busybox-1.33.1.tar.bz2 rootfs
cd rootfs 
tar xvf busybox-1.33.1.tar.bz2
mkrootfs.sh
cd -

wget https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x/linux-5.12.4.tar.xz
tar xvf linux-5.12.4.tar.x

