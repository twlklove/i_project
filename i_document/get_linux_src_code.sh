#uboot:
   git clone https://source.denx.de/u-boot/u-boot.git

#buildroot:
   git clone git://git.busybox.net/buildroot
#   git clone https://git.busybox.net/buildroot

#busybox
wget https://busybox.net/downloads/busybox-1.35.0.tar.bz2


#glibc
#Checkout the latest glibc in development:
git clone https://sourceware.org/git/glibc.git
cd glibc
git checkout master

#Checkout the latest glibc 2.36 stable release:
git clone https://sourceware.org/git/glibc.git
cd glibc
git checkout release/2.36/master
