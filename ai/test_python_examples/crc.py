import sys
import traceback
import argparse
import pycrc.models as cm
from pycrc.algorithms import Crc

def help():
    models = cm.CrcModels()
    print("types supported are : %s" %(models.names()))

    #m = models.get_params('crc-32')
    #print("crc-32 is ", m) 

def get_crc_value(src_value, crc_type):
    models = cm.CrcModels() 
    m = models.get_params(crc_type)

    crc = Crc(m['width'], m['poly'], m['reflect_in'], m['xor_in'], m['reflect_out'], m['xor_out'])
    value=crc.table_driven(src_value)
    print("%s value is 0x%x" %(crc_type, value))
    

if __name__ == '__main__':
    #if len(sys.argv) == 2 : crc_type  = sys.argv[1]
    #parser.add_argument('-test', '--test', default=1, choices=[2, 3, 4], type=int, help='just for help')
    #parser.add_argument('-test', '--test', action='store_true', help='just for help')
    #parser.add_argument('-test', '--test', action='store_const', const=23, help='just for help')
    #parser.add_argument('-test', '--test', nargs=2, type=int, help='just for help')
    #parser.add_argument('-test', '--test', nargs='*', type=int, help='just for help')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='crc-16')
    parser.add_argument('-v', '--value', default='12345')
    args = parser.parse_args() 

    try:   
        get_crc_value(args.value, args.type)
    except Exception as e:
        print('str(Exception):\t', str(Exception))
        print('repr(e):\t', repr(e))
        print('traceback.print_exc():', traceback.print_exc())
    
        help()
