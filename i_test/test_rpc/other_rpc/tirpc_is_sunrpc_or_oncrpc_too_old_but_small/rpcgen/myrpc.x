一、使用rpcgen工具生成RPC底层骨架
 
1.生成my.x文件，然后在该文件编写以下程序
首先创建文件夹rpc（mkdir rpc），以后的所有文件都放在这个文件夹下。
创建my.x文件(my为文件名，.x为后缀)：vi my.x
在my.x填入下面代码：
#define MY_RPC_PROG_NUM         0x38000010   //程序号

struct my_io_data_s        //定义消息结构
{
    int mtype;
    int len;
    char data[1024];
};

typedef struct my_io_data_s my_io_data_t;

program MY_RPC_PROG { 

    version MY_RPC_VERS1 {
        int MY_RPCC(my_io_data_t) = 1;    /* 过程号 = 1 */
    } = 1;        /* Version number = 1 */

    version MY_RPC_VERS2 {
        my_io_data_t MY_RPCC(my_io_data_t) = 1;    /* 过称号 = 1 */
    } = 2;        /* Version number = 2 */

} = MY_RPC_PROG_NUM;    /* Program number */
这里我创建了两个版本，version1和version2,版本的数量是可以自己定制的，如果你需要一个的话定义一个即可。因为我打算定义一个版本用于SET的操作，一个用于GET操作，所以定义了两个版本。
 
上面使用了RPC语言，我对以上几个特殊名词做一下解释。
每个RPC过程由程序号、版本号和过程号来唯一确定。
RPC版本号：程序号标志一组相关的远程过程，程序号的是有范围的，我们需要在范围内填写程序号。
 
程序号范围
简述
0x00000000 - 0x1FFFFFFF
由Sun公司定义，提供特定服务
0x20000000 - 0x3FFFFFFF
由程序员自己定义，提供本地服务或用于调试
0x40000000 - 0x5FFFFFFF
用于短时间使用的程序，例如回调程序
0x60000000 - 0xFFFFFFFF
保留程序号
 
这里我们使用的范围当然是0x20000000 - 0x3FFFFFFF，我填的是0x38000010。
 
版本号：在version的大括号里我们定义两个我们将要使用的RPC调用函数的类型，比如：
version 1：我们定义了int MY_RPCC(my_io_data_t)，这表明我们以后PRC框架使用的RPC调用函数的函数类型那个将会是：int * my_rpcc_1(my_io_data_t *argp, CLIENT *clnt)
 
version 2：my_io_data_t MY_RPCC(my_io_data_t) 则会变成 my_io_data_t * my_rpcc_2(my_io_data_t *argp, CLIENT *clnt)
