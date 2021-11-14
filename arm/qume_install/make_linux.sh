
cd linux-5.12.4
#if [ ! -d  ../rootfs/install_arm64 ]
#then
#    cp ../rootfs/install_arm64 ./ -a
#fi	

#add the content to arch/arm64/configs/defconfig
#add hotplug support
#CONFIG_UEVENT_HELPER=y
#CONFIG_UEVENT_HELPER_PATH="/sbin/hotplog"

#add initramfs supoort
#CONFIG_INITRAMFS_SOURCE="install_arm64"
#vim arch/arm64/configs/defconfig
export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-
make clean
make defconfig
make all -j8

