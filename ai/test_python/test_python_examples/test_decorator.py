from functools import wraps
def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated
 

from functools import wraps
class logit(object):
    def __init__(self, logfile='out.log'):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + ' was called, and write to ' + self.logfile
            print(log_string)
            #with open(self.logfile, 'a') as opened_file:
            #    opened_file.write(log_string + '\n')
            self.notify()
            return func(*args, **kwargs)
        return wrapped_function

    def notify(self):
        pass

class email_logit(logit):
    def __init__(self, email='xx@163.com', *args, **kwargs):
        self.email = email
        super(email_logit, self).__init__(*args, **kwargs)

    def notify(self):
        pass

if __name__ == '__main__':
    @decorator_name
    def func():
        return("Function is running")
     
    can_run = True
    print(func, func.__name__)
    print(func())
     
    can_run = False
    print(func())
    
    @logit()
    def func1():
        pass
   
    print(func1())

    @email_logit()
    def func2():
        pass
    print(func2())
 
