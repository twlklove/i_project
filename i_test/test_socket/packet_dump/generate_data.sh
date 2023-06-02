#!/bin/bash
src_file=tcpdump.data
dst_file=packet.data

cat ${src_file} | sed -E 's/[0-9]{3,}  //g' |  sed -E 's/[ ]{2,}.*//g'| sed -E 's/([0-f]{2})/0x\1,/g' | sed 's/^ /    /g' | sed '1i unsigned char packet_data[ ] = {' | sed '$a };' > ${dst_file}
