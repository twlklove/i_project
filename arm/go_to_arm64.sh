TMOUT=0
ssh -t 192.168.112.129 "TMOUT=0; pwd; ls /home/i_work; cd /home/i_work/qemu; ./start_qemu.sh linux-5.12.4"
