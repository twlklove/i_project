export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/x86_64-linux-gnu/pkgconfig:/root/.local/lib/pkgconfig
export PATH=$PATH:/root/.local/bin
make clean
make -j8

