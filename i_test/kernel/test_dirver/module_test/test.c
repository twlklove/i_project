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

/* debug, name, valid_value is in dir /sys/module/test/parameters/ */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "0:info, 1:all");

static char *name = "hello";
module_param(name, charp, 0644);

static int valid_value[3] = {0,1,2};
int value_num = sizeof(valid_value)/sizeof(valid_value[0]);
module_param_array(valid_value, int, &value_num, 0644);
MODULE_PARM_DESC(valid_value, "valid value");


MODULE_VERSION(DRV_VERSION);
MODULE_AUTHOR("twlklove");
MODULE_DESCRIPTION("a test module");
//MODULE_LICENSE("GPL");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS("test");

/* test_dev_attr is in dir /sys/class/test_class/test_dev/  */
int test_dev_attr[3] = {10, 20, 30};
static ssize_t test_dev_attr_show(struct device* dev, struct device_attribute* attr, char* buf) 
{ 
    return sprintf(buf, "%d %d %d\n", test_dev_attr[0], test_dev_attr[1], test_dev_attr[2]);
}

static ssize_t test_dev_attr_store(struct device* dev, struct device_attribute* attr, const char* buf, size_t count) 
{
    int num = sscanf(buf, "%d %d %d", &test_dev_attr[0], &test_dev_attr[1], &test_dev_attr[2]);
    if (num != 3) {
        printk(KERN_ERR "write num is %d, count is %lu, data is %d, %d, %d\n", num, count, test_dev_attr[0], test_dev_attr[1], test_dev_attr[2]);
        return -EINVAL;
    }

    printk("hi write num is %d, data is %d, %d, %d\n", num, test_dev_attr[0], test_dev_attr[1], test_dev_attr[2]);

    return count;
}

static DEVICE_ATTR_RW(test_dev_attr);

dev_t devt = 1;
struct class *class_test = NULL;
struct device *dev = NULL; 

static __init int test_init(void)
{
    int i = 0;
    printk(KERN_INFO "init\n");

    class_test = class_create(THIS_MODULE, "test_class");
    dev = device_create(class_test, NULL, devt, NULL, "test_dev");
    device_create_file(dev, &dev_attr_test_dev_attr);
 
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
    device_remove_file(dev, &dev_attr_test_dev_attr);
    device_destroy(class_test, devt);
    class_destroy(class_test);

    printk(KERN_INFO "exit\n");
}

module_init(test_init);
module_exit(test_exit);

