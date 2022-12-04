#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    int count = 0;
    printf("input count (<=64) :");
    scanf("%d", &count);
    if ((count <=0) || (count > 64)) {
	printf("should be <= 64\n");
	return -1;
    }

    printf("count is %d\n", count);

    char words[64][17] = {0};
    printf("input %d words (len<=16) :", count);
    int num = 0;
    while(num < count)
    {
        scanf("%s", words[num]);
        printf("words is %s\n", words[num]);
	num++;
    }

    char input_str[513] = {0};
    printf("input centence (len<=512) :");
    scanf("%s", &input_str);
    printf("centence is %s\n", input_str);


    int index = 0;
    char *save_p;
    char *p[1024];
    num = 0;
    for (index = 0; index < count; index++)
    {
        char *str = input_str;
        p[num] = strtok_r(str, words[index], &save_p);
        while(p[num])
        {
            printf("%s ",p[num]);
	    num++;
            p[num]=strtok_r(NULL, words[index], &save_p);
        }
    }

    printf("finish, %s\n", input_str);

    return 0;
}
