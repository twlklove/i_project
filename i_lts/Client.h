#ifndef CHATROOM_CLIENT_H
#define CHATROOM_CLIENT_H

#include <string>
#include "Common.h"

using namespace std;

class Client {
    public:
        Client();
        void Connect();
        void Close();
        void Start();
    private:
        int sock;
        int pid;
        int epfd;
        // 创建管道，其中fd[0]用于父进程读，fd[1]用于子进程写
        int pipe_fd[2];
        // 表示客户端是否正常工作
        bool isClientwork;
        
        // 聊天信息
        Msg msg;
        //结构体要转换为字符串
        char send_buf[BUF_SIZE];
        char recv_buf[BUF_SIZE];
        //用户连接的服务器 IP + port
        struct sockaddr_in serverAddr;
};

#endif
