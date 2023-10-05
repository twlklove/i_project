'''
import queue

q = queue.Queue(maxsize=5)
#q.empty()
print('if q is empty: ', q.empty())

for i in range(4) :
    q.put(i)

print('if q is full: ', q.full())
print('queue size : ', q.qsize())

#q.put(5)
print('if q is full: ', q.full())

q.put_nowait(6)
q.put(7, block=True, timeout=2)
print('queue size : ', q.qsize())

for i in range(4):
    print(q.get())

print(q.get(block=True, timeout=2))
print(q.get(block=False))
'''
'''
import queue
q = queue.Queue()
q.put(1)
q.put(1)
q.put(1)
print('get item: ', q.get())
q.task_done()
print('get item: ', q.get())
q.task_done()
print('get item: ', q.get())

q.join()
'''
'''
from queue import Queue
from threading import Thread

class producer(Thread):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        self.count=5
        while self.count > 0:
            if self.count == 1:
                self.count -= 1
                self.q.put(2)
            else:
                self.count -= 1
                self.q.put(1)

class consumer(Thread):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        while True:
            data = self.q.get()
            if data == 2:
                print("stop because data=", data)
                self.q.task_done()
                break
            else :
                print('data is good, data =', data)
                self.q.task_done()

def main():
    qq = Queue()
    p = producer(qq)
    c = consumer(qq)
    p.setDaemon(True)
    c.setDaemon(True)
    p.start()
    c.start()
    qq.join()
    print("queue is complete")

if __name__ == '__main__':
    main()
'''
'''
import queue
q = queue.LifoQueue()
q.put(1)
q.put(2)
q.put(3)
print(q.get())
print(q.get())
print(q.get())
'''

import queue
q1 = queue.PriorityQueue()
q1.put((1, 'alex1‘))
q1.put((2, 'alex2'))
q1.put((1, 'alex3'))

print(q1.get())
print(q1.get())
print(q1.get())
