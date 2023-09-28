#/bin/bash

start_line=$1
end_line=$2
file=$3

head -n ${end_line} ${file} | tail -n +${start_line}
