import numpy as np
import sys

base_value=1.18
step = 0.01
max_step_count = 13
max_year = 10
width = 6
argc = len(sys.argv)
if argc > 1 :
    max_step_count = int(sys.argv[1])

if argc > 2 :
    max_year = int(sys.argv[2])

for i in range(max_year):
    print('%6d' %(i+1), end=' ')
	
print()

values = [ base_value + i * step for i in range(max_step_count) ]
for i in values:
    for j in range(max_year):
        print('%6.2f' %(np.power(i, j+1)), end=' ')
    print()
