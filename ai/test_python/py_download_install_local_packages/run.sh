
file=requirements.txt
dir=packages
mkdir $dir
tmp_file=tmp
missing_file=missing.txt

rm -rf ${missing_file}
pip freeze > $file
while [ 1 ]
do
    pip download -d $dir -r $file > ${tmp_file} 2>&1
    key=`cat ${tmp_file} | grep "Could not find" | awk -F" " '{print $NF}' | awk -F"==" '{print $1}'`
    if [ "Y${key}" != "Y" ]
    then 
        echo ${key} >> ${missing_file}
        sed -i '/'"${key}"'/d' $file 
    else
        break
    fi
done

pip install --no-index --find-links=${dir}   -r  ${file}

rm -rf ${tmp_file}
