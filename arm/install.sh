cd arm_cross_compile/arm_compile_tool
download.sh
cd -

cd uboot/
download.sh
run.sh
cd -

cd qume_install/
./install_linux_tool.sh
./install_qemu.sh
./make_linux.sh
./create_lib.sh  
./mkrootfs.sh
cd -
