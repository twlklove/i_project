#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include"cJSON.h"

int main(void)
{
    FILE *fp = fopen("create.json","r");
    char string[300] = {0};
    fread(string, sizeof(string), 1, fp);
    //char *string = "{\"family\":[\"father\",\"mother\",\"brother\",\"sister\",\"somebody\"]}";
    fclose(fp);
    printf("%s\n", string);

    //从缓冲区中解析出JSON结构
    cJSON *json = cJSON_Parse(string);
    cJSON *node = NULL;
        
    //判断是否有key是string的项 如果有返回1 否则返回0
    if(1 == cJSON_HasObjectItem(json,"info"))
    {
        printf("found family node\n");
    }
    else
    {
        printf("not found family node\n");
    }

    node = cJSON_GetObjectItem(json,"info");
    if(node->type == cJSON_Array)
    {
        printf("array size is %d\n",cJSON_GetArraySize(node));
    }

    cJSON *tnode = NULL;
    int size = cJSON_GetArraySize(node);
    int i;
    for(i=0;i<size;i++)   // as is :cJSON_ArrayForEach(tnode,node)
    {
        tnode = cJSON_GetArrayItem(node,i);

        if(tnode->type == cJSON_String)
        {
            printf("value[%d]:%s\n",i,tnode->valuestring);
        }
        if(tnode->type == cJSON_Object) {
            if (1 == cJSON_HasObjectItem(tnode,"name")){
                cJSON *p_node = cJSON_GetObjectItem(tnode,"name");
                printf("name is %-8s, ", p_node->valuestring);
            }

            if (1 == cJSON_HasObjectItem(tnode,"age")){
                cJSON *p_node = cJSON_GetObjectItem(tnode,"age");
                printf("age is %-2d, ", p_node->valueint);
            }

            if (1 == cJSON_HasObjectItem(tnode,"address")){
                cJSON *p_node = cJSON_GetObjectItem(tnode,"address");
                printf("address is %-8s\n", p_node->valuestring);
            }
        }
        else
        {
            printf("not find string, type is %d\n", tnode->type);
        }
    }

    cJSON_Delete(json);
    return 0;
}
