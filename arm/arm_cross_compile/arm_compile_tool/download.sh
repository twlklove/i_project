
base_url="https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/"

aarch64_gcc_tool_for_CA=gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu.tar.xz 
arm_gcc_tool_for_CA=gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabi.tar.xz
arm_gcc_tool_for_CM_CR_NO_OS=gcc-linaro-7.4.1-2019.02-x86_64_arm-eabi.tar.xz
files=(${aarch64_gcc_tool_for_CA} ${arm_gcc_tool_for_CA} ${arm_gcc_tool_for_CM_CR_NO_OS})

download_dir=arm_toolchain
install_dir="/usr/local/toolchain"
dirs=(${download_dir} ${install_dir})

for dir in ${dirs[*]} 
do
    if [ -d ${dir} ]
	then
	    continue
	fi

    mkdir -p ${dir} > /dev/null 2>&1
done


bin_path=
for file in ${files[*]}
do
    file_tmp=`echo ${file} | sed 's/.xz//g'`
    if [ ! -f ${download_dir}/${file} ] && [ ! -f ${download_dir}/${file_tmp} ]
	then
	    wget ${base_url}/${file} -P ${download_dir}
	fi

	if [ ! -f ${download_dir}/${file_tmp} ]
	then
	    xz -d ${download_dir}/${file}
	fi

    sub_bin_path=`echo ${file_tmp} | sed 's/.tar//g'`
	if [ -d ${install_dir}/${sub_bin_path}/bin ]
	then
	    continue
	fi

	tar -xvf ${download_dir}/${file_tmp} -C ${install_dir}
	bin_path="${bin_path}:${install_dir}/${sub_bin_path}/bin"    
done


env_file="/root/.bashrc"

if [ "Y${bin_path}" != "Y" ]
then
    path_value=`cat ${env_file} | grep PATH`
	if [ $? -eq 0 ]
	then
        path_value=${path_value}:${bin_path} 
	else
	    path_value="PATH=\${PATH}${bin_path}"
	fi

    sed -i '/PATH\=/d' ${env_file}
    echo $path_value >> ${env_file}	
fi

source ${env_file}

