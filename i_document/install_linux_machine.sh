#sudo passwd root
#su root
#/etc/sshd_config PermitRootxx 
#apt remove xx
#apt-cache search xx

apt install -y linux-source
apt install -y linux-headers-$(uname -r)

apt install -y net-tools
apt install -y sshd
apt install -y vim

apt install -y gcc
apt install -y g++
apt install -y cmake
apt install -y build-essential autoconf libtool pkg-config
apt install -y libssl-dev
apt install -y libncurses-dev
apt install -y libncurses5
apt install -y libpython2.7
apt install -y valgrind 

apt install -y w3m w3m-img  #brower
apt install -y curl
apt install -y wget
apt install -y git

apt install -y python3-pip

pip3 install numpy
pip3 install pandas
pip3 install matplotlib

