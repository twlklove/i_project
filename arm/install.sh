
cd arm_cross_compile/arm_compile_tool
./download.sh
cd -

pushd qume_install/
./install_linux_tool.sh
./install_qemu.sh
./make_linux.sh
pushd rootfs
./create_lib.sh  
./mkrootfs.sh
popd
popd
