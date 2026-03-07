## for root
#sudo passwd
#sudo -

apt install -y vim

## for openssh
apt install -y openssh-server
#set PermitRootLogin yes 
vim /etc/ssh/sshd_config 
service ssh starting

######################
apt install -y git
apt install -y linux-source
apt install -y linux-headers-$(uname -r)
apt install -y gcc
apt install -y g++
apt install -y cmake
apt install -y build-essential autoconf libtool pkg-config
apt install -y libssl-dev
apt install -y libncurses-dev
apt install -y libncurses5
apt install -y net-tools
apt install -y valgrind 

apt install -y w3m w3m-img  #brower
apt install -y curl
apt install -y wget
