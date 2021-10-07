#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/ioport.h>
#include <linux/delay.h>

#include <asm/uaccess.h>
#include <asm/irq.h>
#include <asm/io.h>

#define DRV_VERSION "version 1.0 ("__DATE__": "__TIME__")"

static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "0:info, 1:all");

static char *name = "hello";
module_param(name, charp, 0644);

static int valid_value[10] = {0,1,2};
int value_num;
module_param_array(valid_value, int, &value_num, 0644);
MODULE_PARM_DESC(valid_value, "valid value");


MODULE_VERSION(DRV_VERSION);
MODULE_AUTHOR("twlklove");
MODULE_DESCRIPTION("a test module");
//MODULE_LICENSE("GPL");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS("test");

static __init int test_init(void)
{
    printk(KERN_INFO "init\n");

    int i = 0;
    for_each_possible_cpu(i) {
        printk(KERN_INFO "possible cpu %d\n", i);
    }

    for_each_online_cpu(i) {
        printk(KERN_INFO "online cpu %d\n", i);
    }

    for_each_present_cpu(i) {
        printk(KERN_INFO "present cpu %d\n", i);
    }

    return 0;
}


static __exit void test_exit(void)
{
    printk(KERN_INFO "exit\n");
}

module_init(test_init);
module_exit(test_exit);

