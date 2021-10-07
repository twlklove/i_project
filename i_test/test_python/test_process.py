from multiprocessing import Process
import os
import time

def hello(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    print('hello exit')

def run_proc(name):
    p = Process(target=hello, args=('test_hello',))
    print('run_proc will start.')
    p.start()
    p.join()
    time.sleep(3)
    print('run_proc exit')

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start() 
    #p.join()
    print('run_proc end.')
    time.sleep(1000)
