
OPTIONS=
OPTIONS=${OPTIONS}:detect_leaks=1
OPTIONS=${OPTIONS}:halt_on_error=0                 #检测内存错误后继续运行
OPTIONS=${OPTIONS}:verbosity=0:
OPTIONS=${OPTIONS}:handle_segv=1                   #处理段错误
OPTIONS=${OPTIONS}:detect_stack_use_after_return=1 #检查访问指向已被释放的栈空间
OPTIONS=${OPTIONS}:handle_sigill=1                 #处理SIGILL信号
OPTIONS=${OPTIONS}:use_sigaltstack=0
OPTIONS=${OPTIONS}:allow_user_segv_handler=1
OPTIONS=${OPTIONS}:fast_unwind_on_fatal=1
OPTIONS=${OPTIONS}:fast_unwind_on_check=1
OPTIONS=${OPTIONS}:fast_unwind_on_malloc=1
OPTIONS=${OPTIONS}:quarantine_size=4194304         #内存cache可缓存free内存大小4M
OPTIONS=${OPTIONS}:malloc_context_size=15          #显示的调用栈层数为15
OPTIONS=${OPTIONS}:log_path=asan.log
export ASAN_OPTIONS=${OPTIONS}
export LSAN_OPTIONS=exitcode=0:use_unaligned=0:    #设置内存泄露退出码为0，默认情况内存泄露退出码0x16,use_unaligned=0：字节对齐

make clean
make CFLAG=-DDEBUG ASAN_CHECK=1 

./run

export ASAN_OPTIONS= 
