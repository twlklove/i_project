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
sudo apt update
sudo apt upgrade
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

####################### for python ######################
ln -s /usr/bin/python3 /usr/bin/python
apt install -y pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
apt install -y python`python --version | cut -d' ' -f2 | cut -d'.' -f1-2`-venv

# 创建虚拟环境
python3 -m venv i_env
# 激活虚拟环境  #退出: deactivate
source i_env/bin/activate
pip install numpy
pip install pandas
pip install matplotlib
##################################################

######################  for AI ################
# nvidia-smi #look at gpu info
sudo apt install nvidia-cuda-toolkit
pip install torch

# install opencode
curl -fsSL https://opencode.ai/install | bash

# install openclaw
curl -fsSL https://openclaw.ai/install.sh | bash

# install ollama
curl -fsSL https://ollama.com/install.sh | sh

###################################################
#
#open teminal on machine starting
#echo "gnome-terminal &" >> ~/.bashrc
