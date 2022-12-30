file=i_document/linux_readme
content="linux"  #""  #func_name, key_word and so on

if [ $# -eq 2 ]
then
    echo hello
    file=$1
    content=$2
fi

pattern="\(commit\|Author:\|Date:\|${file}\)"
git log --stat | grep ${file} -n20 | grep ${pattern} |grep -v " commit" | sed 's/commit/\r\ncommit/g' | grep ${file} -B 4 --color=always

commit_num=4
buf=(`git log --stat | grep ${file} -n20 | grep ${pattern} |grep -v " commit" | sed 's/commit/\r\ncommit/g' | grep ${file} -B 4 \
     | grep commit | cut -d' ' -f2 | head -n ${commit_num}`)

num=${#buf[*]}
i=${num}; j=${num}; show_line=10
while [ 1 ]
do    
    let i=$i-1
    if [ ${i} -lt 0 ]
    then
        break
    fi

    commit_i=${buf[i]}
    let j=$i-1
    if [ ${j} -ge 0 ]
    then
        commit_j=${buf[j]}
    else
        j=${i}
        commit_j="  "
    fi

    printf "%3s%3s : %30s  %30s\n" ${i}, ${j} ${commit_i} ${commit_j}

    #git diff commit^!, or git show commit, or git diff commit1 commit2 path

    if [ "Y"${content} == "Y" ]
    then
        git diff --color ${commit_i} ${commit_j} ${file} | xargs -i -n 1 echo {}
    else
        git diff --color ${commit_i} ${commit_j} ${file} | grep ${content} -C ${show_line} --color=always
    fi
   
    echo
    echo
done
 
