
function help()
{
    echo "Usage1: $0 start_commit end_commit yourname"   
    echo "For example: $0 91e2bd1ed7e672a39dd78ef986b5960d451b111d bfd85f11ec98af7e641f3e881e21e71d1d33ca3b wangxiaoming"   

    echo "Usage2: $0 -t start_time end_time yourname"   
    echo "For example: $0 -t 2023-08-25 2023-09-05 wangxiaoming"   

}

author=""

if [[ $# -eq 1 && "Y$1"=="Y-h" ]]; then
    help 
    exit 0
elif [ $# -eq 3 ]; then
    start_commit=$1
    end_commit=$2 
    author=$3
elif [[ $# -eq 4 && "Y$1"=="Y-t" ]]; then
    start_time=$2
    end_time=$3
    author=$4
else
    echo "wront parameters"
    help
    exit -1
fi

if [ "Y$1" == "Y-t" ]; then
lines=`git log --numstat --stat --since=${start_time} --until=${end_time} | cat | grep -E "^commit |Merge|^Author:|Date:|^[0-9-]+|\|[ ]+[0-9+-]+|files changed|file changed" | sed ':a;N;$!ba;s/\n/~~/g'`
else
lines=`git log --numstat --stat ${start_commit}..${end_commit} | cat | grep -E "^commit |Merge|^Author:|Date:|^[0-9-]+|\|[ ]+[0-9+-]+|files changed|file changed" | sed ':a;N;$!ba;s/\n/~~/g'`
fi

total_changes=0
total_insertions=0
total_deletions=0
changes=0
insertions=0
deletions=0
exclude_changes=0
exclude_insertions=0
exclude_deletions=0 

commit_tmp=""
process_marker=0

echo "${lines}" | sed 's/~~/\n/g' | while read -r line
do 
    if [ "Y${author}" != "Y" ]; then
        if [ "Y${process_marker}" != "Y1" ]; then 
            commit_tmp=`echo ${line} | grep "^commit "` 
            if [ "Y$commit_tmp" != "Y" ];then
                process_marker=1
                continue
            fi
        elif [ "Y${process_marker}" == "Y1" ]; then  
            author_tmp=`echo ${line} | grep "^Author:.*${author}"`
            if [ "Y${author_tmp}" != "Y" ]; then
                process_marker=2
                echo ${commit_tmp}
                let exclude_changes=0
                let exclude_insertions=0
                let exclude_deletions=0 
            else 
                process_marker=3
            fi
        fi

        if [ "Y${process_marker}" != "Y2" ]; then
            continue
        fi 
    fi

    content=`echo ${line} | sed 's/[ ]\+/ /g' |grep -E "^[0-9-]+[ ]+[0-9-]+" | grep -v " files changed\| file changed"` 
    if [ "Y${content}" != "Y" ]; then 
       content=`echo ${line} | grep -E "(*\.cpp|*\.h|*\.c|*\.ini|*\.xlsm|*\.bat|*\.sh)$"`
       if [ "Y${content}" != "Y" ]; then
           continue
       fi
      
       num_stat=($(echo ${line} | awk '{print $1, $2}'))
       if [ "Y${num_stat[0]}" != "Y-" ]; then   
           let exclude_insertions=${exclude_insertions}+${num_stat[0]}
           let exclude_deletions=${exclude_deletions}+${num_stat[1]}  
       fi

       let exclude_changes=${exclude_changes}+1
       continue 
    fi

    content=`echo ${line} | grep -E "^commit |Merge|^Author:|Date:|*\.cpp |*\.h |*\.c |*\.ini |*\.xlsm |*\.bat |*\.sh |files changed|file changed"`
    if [ "Y${content}" == "Y" ]; then  
        continue
    fi

    echo $line

    result=`echo ${line} | grep "^[0-9]\+ file"` 
    if [ "Y${result}" != "Y" ]; then
        changes=`echo ${line} | sed 's/[^0-9\,+-]//g'| cut -d',' -f1`
        insertions=`echo ${line} | sed 's/[^0-9\,+-]//g'| cut -d',' -f2 | grep "+" | sed 's/+//g'`
        deletions=`echo ${line} | sed 's/[^0-9\,+-]//g'| cut -d',' -f3 | grep "-" | sed 's/-//g'`
   
        if [ "Y${deletions}" == "Y" ]; then
            if [ "Y${insertions}" == "Y" ]; then
               insertions=0
               deletions=`echo ${line} | sed 's/[^0-9\,+-]//g'| cut -d',' -f2 | grep "-" | sed 's/-//g'`
            else 
               deletions=0
            fi
        fi
         
        let total_changes=${total_changes}+${changes}-${exclude_changes}
        let total_insertions=${total_insertions}+${insertions}-${exclude_insertions}
        let total_deletions=${total_deletions}+${deletions}-${exclude_deletions}
        echo -e "\033[31m==>project file: ${exclude_changes} changed,  ${exclude_insertions} insertions(+), ${exclude_deletions} deletions(-) \033[0m"
        echo -e "\033[32m==>${total_changes} files changed, ${total_insertions} insertions(+), ${total_deletions} deletions(-) \033[0m"

        let exclude_changes=0
        let exclude_insertions=0
        let exclude_deletions=0 
    fi 
done


