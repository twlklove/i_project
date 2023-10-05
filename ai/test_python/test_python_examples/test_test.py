import time
import functools
from inspect import currentframe, getframeinfo

debug=0
text = [debug, 'hello', 'debug']
def time_decorator(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            t1 = time.time()
            func(*args, **kw)
            t2 = time.time()
            if text[0] == 1:
                frameinfo = getframeinfo(currentframe())
                print("%s %s %s %d %f" % (text[1], __file__, func.__name__, frameinfo.lineno, t2 - t1))
        return wrapper
    return decorator

@time_decorator(text)
def test():
    time.sleep(1)

test()