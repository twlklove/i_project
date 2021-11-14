cd u-boot
make clean
make V=1 ARCH=aarch64 CROSS_COMPILE=arm-linux-gnueabi- rpi_3_defconfig
make -j32

cd -

