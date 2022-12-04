 #include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include"cJSON.h"

int main(void)
{
    //先创建空对象
    cJSON *json = cJSON_CreateObject();

    cJSON_AddStringToObject(json,"company","tx");

    //添加数组
    cJSON *array = NULL;
    cJSON_AddItemToObject(json,"info",array=cJSON_CreateArray());

    cJSON *obj = NULL;
    cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
    cJSON_AddItemToObject(obj,"name",cJSON_CreateString("jim"));
    cJSON_AddItemToObject(obj,"age",cJSON_CreateNumber(20));
    cJSON_AddStringToObject(obj,"address","beijing");

    cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
    cJSON_AddItemToObject(obj,"name",cJSON_CreateString("andy"));
    cJSON_AddItemToObject(obj,"age",cJSON_CreateNumber(21));
    cJSON_AddItemToObject(obj,"address",cJSON_CreateString("HK"));
    
    cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
    cJSON_AddStringToObject(obj,"name","eddie");
    cJSON_AddNumberToObject(obj,"age", 22);
    cJSON_AddStringToObject(obj,"address","TaiWan");
    
    //清理工作
    FILE *fp = fopen("create.json","w");
    char *buf = cJSON_Print(json);
    fwrite(buf,strlen(buf),1,fp);
    fclose(fp);
    free(buf);
    cJSON_Delete(json);
    return 0;
}

