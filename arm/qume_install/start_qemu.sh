
dir=linux-5.12.4
if [ $# -eq 1 ]
then
    dir=$1
fi

cd ${dir}
if [ ! -d kmodules ]
then
    mkdir kmodules   # share dir
fi

rootfs_file="../rootfs/rootfs_ext4.img"
if [ -f ${rootfs_file} ]
then
    echo "use rootfs"
    qemu-system-aarch64 -machine virt -cpu cortex-a57 -machine type=virt -m 1024 -smp 4 -kernel arch/arm64/boot/Image \
    	--append "noinitrd root=/dev/vda rw console=ttyAMA0 loglevel=8" -nographic \
    	-drive if=none,file=${rootfs_file},id=hd0 -device virtio-blk-device,drive=hd0 \
    	--fsdev local,id=kmod_dev,path=$PWD/kmodules,security_model=none \
    	-device virtio-9p-device,fsdev=kmod_dev,mount_tag=kmod_mount

else   
    qemu-system-aarch64 -machine virt -cpu cortex-a57 -machine type=virt -m 1024 -smp 4 -kernel arch/arm64/boot/Image \
    	--append "rdinit=/linuxrc root=/dev/vda rw console=ttyAMA0 loglevel=8" -nographic \
    	--fsdev local,id=kmod_dev,path=$PWD/kmodules,security_model=none \
    	-device virtio-9p-device,fsdev=kmod_dev,mount_tag=kmod_mount
fi	

cd -
