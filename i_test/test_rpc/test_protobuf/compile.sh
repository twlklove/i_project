export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/root/.local/lib/pkgconfig
export PATH=$PATH:/root/.local/bin
make clean
make -j8

