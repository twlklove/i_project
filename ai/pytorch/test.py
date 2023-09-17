import sys
import quick_start

is_train=0
if len(sys.argv) == 2 :
    is_train = int(sys.argv[1])

quick_start.main(is_train)

