#include <stdio.h>      /* printf */
#include <stdlib.h>     /* qsort */

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int main ()
{
  int values[] = { 40, 10, 100, 90, 20, 25 };
  qsort (values, sizeof(values)/sizeof(values[0]), sizeof(int), compare);

  int n = 0;
  for (n=0; n<6; n++)
     printf ("%d ",values[n]);
  printf("\n");

  return 0;
}
