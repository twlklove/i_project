echo $0, $#, pid is : $$
father_caller=`ps aux | grep $0`
logger -s $father_caller                  #echo info into messages

src_dir=$(cd $(dirname $0); pwd)/../../main
cppcheck -j 8 --enable=all --force --check-config --inconclusive   ${src_dir}/*.c

