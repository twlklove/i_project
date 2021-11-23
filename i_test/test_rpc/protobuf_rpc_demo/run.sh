apt install -y boost
apt install -y libboost
apt install libboost-all-dev

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/x86_64-linux-gnu/pkgconfig:/root/.local/lib/pkgconfig
export PATH=$PATH:/root/.local/bin
cd src
protoc -I . --cpp_out=.  rpc_meta.proto 
protoc -I . --cpp_out=.  echo.proto 
cd -

mkdir tmp
cd tmp
cmake ../
make -j9
cd -
 
