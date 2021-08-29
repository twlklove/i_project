#!/bin/bash

mkdir cpuset
mount -t cpuset cpuset cpuset
cd cpuset/
mkdir prodset
cd prodset/
echo 4-7 > cpuset.cpus
echo 1 > cpuset.cpu_exclusive
echo 0 > cpuset.mems
echo 36764 > tasks

