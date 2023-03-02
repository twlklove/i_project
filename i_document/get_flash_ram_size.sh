#!/bin/bash

elf_read=arm-none-eabi-readelf
elf_flag=-S
cur_dir=`pwd`
build_dir=${cur_dir}/build 
elf_file=${build_dir}/RTOSDemo.axf    #execute elf file, such as vmlinux

####
target_file=${build_dir}/run.bin    #bin file for machine run, such as Image
arm-none-eabi-objcopy -O binary -R .note -R .comment -S ${elf_file} ${target_file}  # convert axf or elf to bin
ls ${build_dir}/* -l 

#####################
#Flash: .isr_vector|.text|.data|.uninitialized|.comment|.ARM.attributes|.debug_*|..symtab|.strtab|.shstrtab 
echo Flash is:
${elf_read} ${elf_flag} ${elf_file} | grep -E "(Nr|.isr_vector|.text|.data|.uninitialized)"
echo

#RAM:   .data|.uninitialized|.bss|.heap|.stack
echo RAM is:
${elf_read} ${elf_flag} ${elf_file} | grep -E "(Nr|.interrupts_ram|.data|.uninitialized|.bss|.heap|.stack)"


#elf_flag=-a
#elf_section_s="Section Headers:"
#elf_section_e="Key to Flags:"
#${elf_read} ${elf_flag} ${elf_file} | grep -Pzo "(?s)${elf_section_s}.*?${elf_section_e}"; echo


