/* bsearch example */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* qsort, bsearch, NULL */

int compareints (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int main ()
{
  int values[] = { 50, 20, 60, 40, 10, 30 };
  qsort (values, sizeof(values)/sizeof(values[0]), sizeof(values[0]), compareints);

  int key = 40;
  int *pItem = (int*)bsearch(&key, values, sizeof(values)/sizeof(values[0]), sizeof(values[0]), compareints);
  if (pItem!=NULL)
    printf ("%d is in the array.\n",*pItem);
  else
    printf ("%d is not in the array.\n",key);

  return 0;
}
