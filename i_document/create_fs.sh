#1.创建空文件
dd if=/dev/zero of=rootfs.ext4 bs=1M count=32

#格式化为ext4文件系统
mkfs.ext4 rootfs.ext4

if [ $# -ne 1 ]
then
    echo no src dir
    exit
else 
    src_dir=$1
fi

#拷贝文件
mkdir mnt
mount rootfs.ext4 mnt/
cp -rf ${src_dir}/* mnt/
umount mnt
rm -rf mnt
