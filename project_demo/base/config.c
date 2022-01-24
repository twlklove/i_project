#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "config.h"
#include "cJSON.h"
#include "log.h"

#define MAX_CFG_FILE_SIZE 3000
#define CFG_FILE "cfg.json"

int parse_log_cfg()
{
    int ret = 0;
    char cfg[MAX_CFG_FILE_SIZE] = {0};

    FILE *fp = fopen(CFG_FILE, "r");
    if (NULL == fp) {
        DUMP(err, "fail to open config file\n");
        return -1;
    }

    fread(cfg, MAX_CFG_FILE_SIZE-1, 1, fp);
    fclose(fp);
    
    cJSON *json = cJSON_Parse(cfg);
        
    do {
        if(1 != cJSON_HasObjectItem(json, "config")){
            DUMP(err, "not found config node\n");
            ret = 1;
            break;
        }

        if(1 != cJSON_HasObjectItem(json, "log_cfg")){
            DUMP(err, "not found log_cfg node\n");
            ret = 1;
            break;
        }

        cJSON *p_node = cJSON_GetObjectItem(json, "log_cfg");
        if(p_node->type == cJSON_Array){
            cJSON *p_c_node = NULL;
            cJSON_ArrayForEach(p_c_node, p_node) {
                if(p_c_node->type == cJSON_Object) {
                    if (1 == cJSON_HasObjectItem(p_c_node,"level")){
                        cJSON *p_node_tmp = cJSON_GetObjectItem(p_c_node,"level");
                        DUMP(debug, "level is %-8s\n", p_node_tmp->valuestring);
                        set_dump_level(p_node_tmp->valuestring);
                    }

                    if (1 == cJSON_HasObjectItem(p_c_node,"dump_to")){
                        cJSON *p_node_tmp = cJSON_GetObjectItem(p_c_node,"dump_to");
                        DUMP(debug, "dump_to is %-2d\n", p_node_tmp->valueint);
                        set_dump_to_file(p_node_tmp->valueint);
                    }
                }
            }
        }
        else {
            DUMP(err, "not find log_cfg");
            break;
        }
    } while(0);

    cJSON_Delete(json);
    return ret;
}
