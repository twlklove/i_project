src_dir=.

if [ $# -eq 1 ]
then
    src_dir=$1
fi

find $(src_dir) | cpio -o -H newc > rootfs.cpio
gzip -c rootfs.cpio > rootfs.cpio.gz

