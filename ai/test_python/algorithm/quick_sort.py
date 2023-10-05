
def quick_sort(arr):
    length = len(arr)
    if length < 1 :
        return arr

    middle_index = length // 2 
    middle_value = arr[middle_index]

    left_values=list()
    right_values=list()
    middle_values=list()

    for i in range(length) :
        if arr[i] < middle_value :
            left_values.append(arr[i])
        elif  arr[i] > middle_value :
            right_values.append(arr[i])
        else :
            middle_values.append(arr[i])
            
    return quick_sort(left_values) + middle_values + quick_sort(right_values)

print(quick_sort([1,5,10, 20, 4,3,2,0]))



