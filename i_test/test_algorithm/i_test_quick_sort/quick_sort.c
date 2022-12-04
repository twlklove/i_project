#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int partition(int *ids, int p, int r) 
{
    int value = ids[r];
    int tmp = 0;
    int i = p - 1;
    int j = p;
    for (; j <= r - 1; j++) {
        if (ids[j] <= value) {
            i = i + 1;
            tmp = ids[i];
            ids[i] = ids[j];
            ids[j] = tmp;
        }
    }

    tmp = ids[i + 1];
    ids[i+1] = value;
    ids[r] = tmp;
    return i+1;
}

void quick_sort(int *ids, int p, int r) 
{
    if (p < r) {
        int q = partition(ids, p, r);
        quick_sort(ids, p, q-1);
        quick_sort(ids, q+1, r);
    }    
}

#ifdef DEBUG
void test_0()
{
    int ids[] = {1, 100, 9, 5, 7, 2, 8, 3};
    int len = sizeof(ids)/sizeof(ids[0]); 

    quick_sort(ids, 0, len-1);

    int i = 0;
    for (i = 0; i < len; i++) {
        printf("%d ", ids[i]);
    }

    printf("\n");
}

int main() 
{
    test_0();
    return 0;
}
#endif
