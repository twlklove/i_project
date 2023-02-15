#include <stdio.h>
 
void do_shell_cmd(char *p_cmd, char *p_result)
{
    if (NULL == p_cmd) {
        return;
    }

    int loop = 0;
    FILE *fp = NULL;
    char buf[100]={0}; 
    char *p_buf = p_result;
    if (NULL == p_result) {
        loop = 1;
        p_buf = buf;
    }

    fp = popen(p_cmd, "r");
    if (fp) {
        do { 
            int ret =  fread(p_buf, 1,sizeof(buf)-1,fp);
            if(ret > 0) {
                printf("%s",buf);
                continue;
            } 

            break;
        } while (loop);
    }
    
    if (fp) {
        pclose(fp);
    }
    
    printf("\n");
}

#ifdef TEST
int main(int argc, char *argv[])
{
    char cmd[100] = {0};
    sprintf(cmd, "%s", "ps -ef");
    do_shell_cmd(cmd, NULL);

    printf("\nhello\n");
    char p_buf[100] = {0};
    sprintf(cmd, "%s", "ps -ef | grep ksoftirqd | grep -v grep");
    do_shell_cmd(cmd, p_buf);
    printf("%s\n", p_buf);

    return 0;
}
#endif
