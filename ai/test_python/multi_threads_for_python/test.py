import threading
import time

'''
def worker():
    time.sleep(2)
    print("test")

for i in range(5):
    t = threading.Thread(target=worker)
    t.start()
'''
'''
class MyThread(threading.Thread):
    def __init__(self, func, args):
        self.func = func
        self.args = args
        super(MyThread, self).__init__()

    def run(self):
        self.func(self.args)

def f2(arg):
    print(arg)

obj = MyThread(f2, 123)
obj.start()
'''
'''
import threading
import time

NUM = 0
class MyThread(threading.Thread):
    def run(self, l):
        l.acquire()
        global NUM
        NUM += 1
        time.sleep(0.5)
        msg = self.name + ' set num to ' + str(NUM)
        l.release()
        print(msg)

if __name__ == '__main__':
    lock = threading.Lock()
    for i in range(5):
        t = MyThread()
        t.run(lock)
'''

import threading, time
def run(n):
    semaphore.acquire()
    time.sleep(1)
    print("run the thread:%s" %n)
    semaphore.release()

if __name__ == '__main__':
    num = 0
    semaphore = threading.BoundedSemaphore(5)
    for i in range(20):
        t = threading.Thread(target=run, args=(i,))
        t.start()
