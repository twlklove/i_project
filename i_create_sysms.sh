#!/bin/bash

if [ $# -ne 2 ]
then
    echo "please input the right parameters"
    exit 1 
else
    echo "input file is $1, output file is $2"
    input=$1
    output=$2
fi

readelf -s ${input} | grep -v "_.*" \
	| grep -E "FUNC | OBJECT" \
	| grep "[0-9]\+[ ].*GLOBAL DEFAULT[ ]\+[0-9]\+"\
       	| awk '{print $4, $8, $2}' \
	| sort  \
	| awk '{$1=($1=="FUNC")?"()":""; print $2$1 " @ " "*0x"$3}' > ${output}
