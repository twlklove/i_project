#include "list.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    struct list_head list;
	u32 type_id;
	u32 len;	
    u32 data[6];
}data_0;

typedef struct {
    struct list_head list;
	u32 type_id;
    u32 len;
    u32 data[0];
}data_1;

typedef struct {
    struct list_head list;
	u32 type_id;
    u32 len;
    u8 *data;
}data_2;


void test_0()
{
    LIST_HEAD(test_list);
    data_0 data0 = {.len=4, .data={0,1,2,3}};
    list_add_tail(&data0.list, &test_list);

    data_0 *pos = NULL;
    list_for_each_entry(pos, &test_list, list) {
	    int i = 0;
	    for (i = 0; i < pos->len; i++) {
	        printf("%d ", pos->data[i]);
	    }
		printf("\n");
	}
}

#define FREE(test_list, pos, last_post) ({ \
	pos = NULL; \
	last_pos = NULL; \
    list_for_each_entry(pos, &test_list, list) { \
	    if (NULL != last_pos) { \
	        list_del(&last_pos->list);	\
	        free(last_pos); \
			last_pos = NULL; \
		} \
		last_pos = pos; \
	} \
	if (NULL != last_pos) {  \
	    list_del(&last_pos->list);	\
	    free(last_pos); \
		last_pos = NULL; \
    } \
}) 


void put_data(struct list_head *test_list, u32 *p_data, u32 data_size)
{
	data_1 *p_data1 =  (data_1*)malloc(sizeof(data_1) + data_size);
	if (NULL == p_data1) {
	    printf("null pointer\n");
	    return;
	}
    
	p_data1->len = data_size;
	p_data1->type_id = 1;
	memcpy(p_data1->data, p_data, data_size);

    list_add_tail(&p_data1->list, test_list);
}

void test_1()
{
    LIST_HEAD(test_list);

    u32 data[] = {1, 2, 3, 4};
    put_data(&test_list, data, sizeof(data));

    u32 data2[] = {2, 1, 2, 3, 4, 5, 6};
    put_data(&test_list, data2, sizeof(data2));

    data_1 *pos = NULL;
    list_for_each_entry(pos, &test_list, list) {
	    int i = 0;
	    for (i = 0; i < pos->len / sizeof(pos->data[0]); i++) {
	        printf("%d ", pos->data[i]);
	    }
		printf("\n");
	}

    data_1 *last_pos = NULL;
	FREE(test_list, pos, last_pos);
}

void test_2()
{
    LIST_HEAD(test_list);
    
	data_2 data2 = {.len=5, .data="hello"};
	list_add_tail(&data2.list, &test_list);

	data_2 data3 = {.len=3, .data="hi"};
	list_add_tail(&data3.list, &test_list);

    data_2 *pos = NULL;
    list_for_each_entry(pos, &test_list, list) {
	    printf("%s\n", pos->data);
	}
}

typedef struct {
   struct list_head list;
	u32 type_id; 
}data_base_type;

typedef struct {
    data_base_type base; 
	u32 len;	
    u32 data[6];
}data_10;

typedef struct {
    data_base_type base;
    u32 len;
    u8 *data;
}data_20;

typedef void (*callback)(data_base_type *pos);

void process_data10(data_base_type *pos) 
{
    data_10 *data_tmp = container_of(pos, data_10, base);   
    int i = 0;
    for (i = 0; i < data_tmp->len; i++) {
        printf("%d ", data_tmp->data[i]);
    }
    printf("\n"); 
}

void process_data20(data_base_type *pos) 
{
    data_20 *data_tmp = container_of(pos, data_20, base);  
	printf("%s\n", data_tmp->data); 
}
callback callbacks[] = {process_data10, process_data20};

void test_3()
{
    LIST_HEAD(test_list);
   
    data_10 data0 = {.len=4, .base.type_id=0, .data={0,1,2,3}};
    list_add_tail(&(data0.base.list), &test_list);

	data_20 data2 = {.len=5, .base.type_id=1, .data="hello"};
	list_add_tail(&(data2.base.list), &test_list);

    printf("%s\n", __FUNCTION__);
	data_base_type *pos;
	list_for_each_entry(pos, &test_list, list) {
	    callbacks[pos->type_id](pos);
	}
}

int main(int argc, char *argv[])
{
    test_0();
    test_1();
    test_2();        
	test_3();
	return 0;
}
