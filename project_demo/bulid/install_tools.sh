wget yum install cppcheck http://sourceforge.net/projects/cppcheck/files/cppcheck/2.6/cppcheck-2.6.tar.gz
tar xvf cppcheck-2.6.tar.gz
cd cppcheck-2.6
mkdir -p /usr/bin/cppcheck
make -j9 FILESDIR=/usr/bin/cppcheck
make install FILESDIR=/usr/bin/cppcheck
#cp cfg your project dir
cp cfg ../ -rf
cd -


git clone https://github.com/DaveGamble/cJSON.git
