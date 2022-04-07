
version=1.0.1  #main_version.sub_version.modify_version (Note:even is salid version, and odd is not)

check_usage()
{
    if [ $# -lt 5 ]
    then
        if [ $1 == "-h" ]
        then
            echo "\$1 is file name; \$2 is row,and 0 is all rows; \$3 is column; \$4 is pattern \$5 is new content"
        elif [ $1 == "-v" ]
        then
            echo "$version"
        else 
            echo "please input right parameters"
        fi  
        exit 0
    fi
}

modify_content_on_row_column()
{
    #获取环境变量：awk  'BEGIN{for (i in ENVIRON) {print i"="ENVIRON[i];}}'
    #awk ‘{action}’  变量名=变量值
    #awk –v 变量名=变量值 [–v 变量2=值2 …] 'BEGIN{action}’
    #awk -F " " '{if ($i=="hello") $i="hi"}1' i="$column" ${file_name} > ${output_file_name} 
    file_name=$1
    row=$2
    column=$3
    src_content=$4
    dst_content=$5

    if [ ! -f ${file_name} ]
    then
	echo "file ${file_name} is not exist"
	exit 1;
    fi 

    output_file_name=${file_name}.out
    dos2unix ${file_name}

    awk -F " " -v row_tmp=$row -v column_tmp=$column -v src=$src_content -v dst=$dst_content \
	    '{ if(((0==row_tmp)||(NR==row_tmp))&&($column_tmp==src)){$column_tmp=dst} }1' ${file_name} > ${output_file_name}
}

main()
{
    check_usage $@
    modify_content_on_row_column $@ 
}

main $@
