busy_box=busybox-1.33.1
cd ${busy_box}
export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-

# Settings --> [*]Build static binary(no shared libs)
make menuconfig

make
make install

#cd etc
#ls

#mkdir dev etc lib sys proc tmp var home root mnt

cd -

#mkdir append_to_install
#cd append_to_install/
#mkdir dev etc lib lib sys proc tmp var home root mnt
#add files to /etc and so on
#cd dev/
#mknod console c 5 1
#cd ../lib/
#cp /usr/aarch64-linux-gnu/lib/*.so* -a .
#aarch64-linux-gnu-strip *

rm -rf install_arm64
cp ${busy_box}/_install install_arm64 -a 
./create_lib.sh
ls append_to_install/ | grep -v bin | xargs -i -n 1 cp append_to_install/{} install_arm64/ -rf
cp append_to_install/bin/* install_arm64/bin/ -rf

rootfs_file=rootfs_ext4.img
rm -rf ${root_file}
dd if=/dev/zero of=${rootfs_file} bs=1M  count=4096
mkfs.ext4 rootfs_ext4.img
mkdir -p tmpfs
mount -t ext4 ${rootfs_file} tmpfs/ -o loop
cp -af install_arm64/* tmpfs/
umount tmpfs
rm -rf tmpfs
chmod 777 ${rootfs_file}
