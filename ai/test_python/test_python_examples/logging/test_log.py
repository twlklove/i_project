#! /usr/bin/env python
import logging,os

#import ctypes 
#FOREGROUND_WHITE = 0x0007
#FOREGROUND_BLUE = 0x01 # text color contains blue.
#FOREGROUND_GREEN= 0x02 # text color contains green.
#FOREGROUND_RED = 0x04 # text color contains red.
#FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

'''
STD_OUTPUT_HANDLE= -11
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
def set_color(color, handle=std_out_handle):
    print('\033[40;' + str(color) + 'm')
    #bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return bool
'''

'''
字背景颜色范围:40–47
40:黑 41:红 42:绿 43:黄色 44:蓝色 45:紫色 46:天蓝 47:白色
字颜色:30–37
30:黑 31:红 32:绿 33:黄 34:蓝色 35:紫色 36:天蓝 37:白色
ANSI控制码的说明：
\33[0m 关闭所有属性 \33[1m 设置高亮度 \33[4m 下划线 \33[5m 闪烁 \33[7m 反显
'''

FOREGROUND_DEFAULT = 0
FOREGROUND_WHITE = 37
FOREGROUND_BLUE = 34 # text color contains blue.
FOREGROUND_GREEN= 32 # text color contains green.
FOREGROUND_RED = 31 # text color contains red.
FOREGROUND_YELLOW = 33

def set_color(color):
    print('\033[40;' + str(color) + 'm', end='')
    return bool

class Logger:
 def __init__(self, path,clevel = logging.DEBUG,Flevel = logging.DEBUG):
  self.logger = logging.getLogger(path)
  self.logger.setLevel(logging.DEBUG)
  fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

  sh = logging.StreamHandler()
  sh.setFormatter(fmt)
  sh.setLevel(clevel)

  fh = logging.FileHandler(path)
  fh.setFormatter(fmt)
  fh.setLevel(Flevel)

  self.logger.addHandler(sh)
  self.logger.addHandler(fh)
 
 def debug(self,message):
  self.logger.debug(message)
 
 def info(self,message):
  self.logger.info(message)
 
 def war(self,message,color=FOREGROUND_YELLOW):
  set_color(color)
  self.logger.warning(message)
  set_color(FOREGROUND_DEFAULT)
 
 def error(self,message,color=FOREGROUND_RED):
  set_color(color)
  self.logger.error(message)
  set_color(FOREGROUND_DEFAULT)
 
 def cri(self,message):
  self.logger.critical(message)
 
if __name__ =='__main__':
 logyyx = Logger('yyx.log',logging.WARNING,logging.DEBUG)
 logyyx.debug('a debug message')
 logyyx.info('a info message')
 logyyx.war('a warning message')
 logyyx.error('a error message')
 logyyx.cri('a critical message')
