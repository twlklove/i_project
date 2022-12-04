code_dir=../../i_opensource
mkdir -p ${code_dir}
cd ${code_dir}

## linux
#git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git  #all versions
#cd linux
#git checkout -b v5.15 #git chekcout v5.15
#git reset v5.15 --hard   # switch to v5.15
#git branch -a -v
#cd -
git clone https://github.com/torvalds/linux.git

## uboot:
   git clone https://source.denx.de/u-boot/u-boot.git

## buildroot:
   git clone git://git.busybox.net/buildroot
#   git clone https://git.busybox.net/buildroot

## busybox
wget https://busybox.net/downloads/busybox-1.35.0.tar.bz2

########### other src code ###############
## glibc
#Checkout the latest glibc in development:
git clone https://sourceware.org/git/glibc.git
cd glibc
git checkout master

##Checkout the latest glibc 2.36 stable release:
#git clone https://sourceware.org/git/glibc.git
#cd glibc
#git checkout release/2.36/master

## openssl
git clone https://github.com/openssl/openssl.git

## opencv
git clone https://github.com/opencv/opencv.git

## dhcp
wget https://ftp.isc.org/isc/dhcp/4.4.3/dhcp-4.4.3.tar.gz

## lwip
#git clone https://github.com/lwip-tcpip/lwip.git

## grpc and tensorflow
git clone https://github.com/grpc/grpc.git
git clone https://github.com/protocolbuffers/protobuf.git
git clone https://github.com/tensorflow/tensorflow.git

cd -
