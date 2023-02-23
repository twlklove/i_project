#sudo passwd root
#su root
#/etc/sshd_config PermitRootxx 
#apt remove xx
apt install -y net-tools
apt install -y sshd
apt install -y vim

apt install -y gcc
apt install -y g++
apt install -y cmake
apt install -y build-essential autoconf libtool pkg-config
apt install -y libssl-dev
apt install -y libncurses-dev
apt install -y valgrind 

apt install -y w3m w3m-img  #brower
apt install -y curl
apt install -y wget
apt install -y git

apt install -y python3-pip

pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install tensorflow
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
#pip3 install torch torchvision torchaudio      # for cuda=11.7
#conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
