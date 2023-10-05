from multiprocessing import Process
import os
import time

def run_proc(name):
    print('run_proc will start.')
    os.system("./test.sh")
    print('run_proc exit')

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start() 
    print(time.time())
    p.join(3)
    print(time.time())
    print('byebye')
    time.sleep(1000)
