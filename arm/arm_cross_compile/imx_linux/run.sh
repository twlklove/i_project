
packages=(openssl openssl-devel)
for pack in ${packages[@]}
do
    rpm -qa | grep ${pack} >/dev/null 2>&1
	if [ $? -eq 0 ]
	then
	    continue
	fi

    yum install ${pack} 
done 

#begin to compile

cd imx_linux
source ~/.bashrc
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-

file arch/arm64/boot/Image.gz 
file arch/arm64/boot/Image
cd -
