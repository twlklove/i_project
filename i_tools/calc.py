import numpy as np
import sys

base_value=1.18
step = 0.01
max_step_count = 13
max_year = 10
width = 6

base_value_index = 1
max_step_count_index = 2
max_year_index = 3

argc = len(sys.argv)
if argc > base_value_index :
    base_value = float(sys.argv[base_value_index])

if argc > max_step_count_index :
    max_step_count = int(sys.argv[max_step_count_index])

if argc > max_year_index :
    max_year = int(sys.argv[max_year_index])

for i in range(max_year):
    print('%6d' %(i+1), end=' ')
	
print()

values = [ base_value + i * step for i in range(max_step_count) ]
for i in values:
    for j in range(max_year):
        print('%6.2f' %(np.power(i, j+1)), end=' ')
    print()
