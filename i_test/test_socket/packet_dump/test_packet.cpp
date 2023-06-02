#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <csignal>

#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <sys/times.h>

#include <sys/socket.h> 
#include <netpacket/packet.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>

#include <netdb.h>             // gethostbyname
#include <arpa/inet.h>         // ntohs ntohl htons htonl

#include <netinet/in.h>        // IPPROTO_TCP
#include <netinet/udp.h>
#include <netinet/tcp.h>

#include <netinet/ip.h>

// arp
#include <netinet/if_ether.h>  
#include <net/if_arp.h>  

// ether
#include <netinet/ether.h>      //ETHERTYPE_IP
#include <net/ethernet.h>
#include <net/if.h>   //ifreq

#include <iostream>
#include <cstring>
#include <vector>
#include <list>
#include <set>
#include <map>

#include "types.h"
#include "log.h"
#include "checksum.h"

#define ETH   ((s8*)("ens33"))

#define DHCP_REQ_PORT     0x44
#define DHCP_RESP_PORT    0x43

/* etherner_packet = 14B ether_hdr + protocol_data + 4B CRC */
#define ETH_MIN_FRAME_LEN  42           //   42B: 14B ether_hdr_len + 28Barp_hdr_len
#define ETH_MAX_FRAME_LEN  ETH_FRAME_LEN  // 1514B: 14B ether_hdr_len + 1500B MTU

//xxyyoooo[proto:subproto:op]
typedef enum {
    PROTO_PACKET,
    PROTO_ARP           =   0x01000001,
    PROTO_ARP_REQ,
    PROTO_ARP_REPLY,

    PROTO_UDP           =   0x02000004,
    PROTO_BROADCAST     =   0x02010005,
    PROTO_DHCP          =   0x02020006,
    PROTO_DHCP_REQ,
    PROTO_DHCP_RESP,
    PROTO_UDP_NORMAL    =   0x02030009,

    PROTO_TCP           =   0x0300000a,

    PROTO_OTHER_MSG     =   0x0400000b, 
}msg_type;
const char *msg_infos[] = {"all packet", "arp", "arp req", "arp reply", "udp", "broadcast", 
                       "dhcp", "dhcp req", "dhcp resp", "udp_normal", "tcp", "other_msg"};

typedef struct {
    msg_type msg_type_id;
}msg_info;

enum 
{
    OPEN_ETH_FLAG,
    CLOSE_ETH_FLAG,
};

//#pragma pack(1)
typedef struct
{
    struct ether_header ether_hdr;
    
    union
    {
        struct 
        {
            struct iphdr __ip_hdr;
            union
            {
                struct tcphdr ___tcp_hdr;
                struct udphdr ___udp_hdr;
            }__trans_hdr;
        }_inet_hdr;

        struct ether_arp _arp_hdr;
    }proto_hdr;

#define p_ether_hdr_dst_mac          ether_hdr.ether_dhost   //ETHER_ADDR_LEN
#define p_ether_hdr_src_mac          ether_hdr.ether_shost
#define ether_hdr_type               ether_hdr.ether_type

#define ip_hdr                       proto_hdr._inet_hdr.__ip_hdr
#define ip_hdr_len                   ip_hdr.ihl
#define ip_tot_len                   ip_hdr.tot_len
#define ip_protocol                  ip_hdr.protocol
#define ip_check                     ip_hdr.check
#define ip_saddr                     ip_hdr.saddr
#define ip_daddr                     ip_hdr.daddr

#define udp_hdr                      proto_hdr._inet_hdr.__trans_hdr.___udp_hdr
#define udp_src_port                 udp_hdr.source
#define udp_dst_port                 udp_hdr.dest

#define tcp_hdr                      proto_hdr._inet_hdr.__trans_hdr.___tcp_hdr
#define tcp_src_port                 tcp_hdr.source
#define tcp_dst_port                 tcp_hdr.dest
#define tcp_seq                      tcp_hdr.seq
#define tcp_ack_seq                  tcp_hdr.ack_seq
#define tcp_hdr_len                  tcp_hdr.doff
#define tcp_syn                      tcp_hdr.syn
#define tcp_ack                      tcp_hdr.ack
#define tcp_wnd                      tcp_hdr.window
#define tcp_check                    tcp_hdr.check
#define tcp_urg_ptr                  tcp_hdr.urg_ptr  // u16

// netinet/if_ether.h [net/ethernet.h  net/if_arp.h]   ETHER_ADDR_LEN  ARPOP_REQUEST ARPOP_REPLY
#define arp_hdr                      proto_hdr._arp_hdr
#define arp_op_cmd                   arp_hdr.arp_op    //#define arp_op ea_hdr.ar_op in netinet/if_ether.h
#define p_arp_src_mac                arp_hdr.arp_sha
#define p_arp_src_ip                 arp_hdr.arp.spa
#define p_arp_dst_mac                arp_hdr.arp_tha
#define p_arp_dst_ip                 arp_hdr.arp_tpa

#define ETH_HEADER_BASE_SIZE              14
#define IP_HEADER_BASE_SIZE               20
#define TCP_HEADER_BASE_SIZE              20 
#define TCP_TLV_INVALID                   0x00
#define TCP_TLV_NOP                       0x01   // no lv
#define TCP_TLV_WINDOW_SCALE              0x03   

#define IS_BROADCAST(p_head)                                                   \
({                                                                             \
    s32 ret = 0;                                                               \
    do {                                                                       \
        u64 dst_mac_0 = ntohl(*((u32*)((p_head)->p_ether_hdr_dst_mac)));         \
        u64 dst_mac_1 = ntohs(*((u16*)((p_head)->p_ether_hdr_dst_mac + 4)));     \
        u64 dst_mac = ((dst_mac_0 << 16) | dst_mac_1);                         \
        if (0xFFFFFFFFFFFF == dst_mac) {                                       \
                ret = 1;                                                           \
        }                                                                      \
    } while(0);                                                                \
    ret;                                                                       \
})

#define IS_NO_IP_EXTEND_HEADER(p_head)                                         \
({                                                                             \
    s32 ret = 0;                                                               \
    do {                                                                       \
        if (IP_HEADER_BASE_SIZE == ((p_head)->ip_hdr_len) * 4) {                   \
            ret = 1;                                                           \
        }                                                                      \
    } while(0);                                                                \
    ret;                                                                       \
})

#define IS_NO_TCP_EXTEND_HEADER(p_head)                                        \
({                                                                             \
    s32 ret = 0;                                                               \
    do {                                                                       \
        if (TCP_HEADER_BASE_SIZE == ((p_head)->tcp_hdr_len) * 4) {                 \
            ret = 1;                                                           \
        }                                                                      \
    } while(0);                                                                \
    ret;                                                                       \
})

}__attribute__((packed)) frame_header_no_hdr_extend;
//#paragma pack()


u32 recv_thread_running = 1;
u32 eth_ip = 0;

s8 *cfg_p_eth_name = ETH;
msg_type cfg_dump_proto = PROTO_PACKET;
u32 cfg_dump_count = 0xFFFFFFFF;
u32 cfg_dump_len = DEFAULT_DUMP_LEN;
const u32 proto_mask = 0xFF000000;
const u32 sub_proto_mask = 0x00FF0000;
const u32 proto_info_mask  = 0x0000FFFF;
u32 cfg_dump_proto_id = 0; 
u32 cfg_dump_sub_proto_id = 0;
u32 cfg_ignore_outgoing = 0;

void test_for_create_new_tcp_packet();
void test_for_checksum();

u16 tcp_v4_check(u32 len, u32 saddr, u32 daddr, u32 csum)
{
    return csum_tcpudp_magic(saddr, daddr, len, IPPROTO_TCP, csum);
}

u32 verify_tcp_checksum(u8 *p_data, u16 data_len, u8 update)
{
    u32 ret = 0;
    frame_header_no_hdr_extend *p_head = (frame_header_no_hdr_extend*)p_data;  
    u32 tcp_len = ntohs(p_head->ip_tot_len) - p_head->ip_hdr_len * 4;
    dump("src port is 0x%04x, dst port is 0x%04x, ip len is %d, tcp len is %d\n", 
           ntohs(p_head->tcp_src_port), ntohs(p_head->tcp_dst_port), ntohs(p_head->ip_tot_len), tcp_len);
 
    /* for send packet, csum is not finally csum, here is csum = ~tcp_v4_check(tcp_len, p_head->ip_saddr, p_head->ip_daddr, csum), 
     * so dont't check here */

    if (eth_ip == ntohl(p_head->ip_saddr)) { 
        dump("[T] ==> a send packet\n");
        return 0;
    }

    dump("[R] ==> a recv packet\n");

    struct tcphdr *p_tcph = (struct tcphdr*)(&(p_head->tcp_hdr));
    u16 src_check = p_head->tcp_check;
    p_head->tcp_check = 0;
    u16 csum = 0;

    // tcp_check = tcp_head_check + tcp_payload_check + 12B_tcp_fake_header_check  
    csum = csum_partial(p_tcph, tcp_len, csum);

    // 12B tcp_fake_header_check : src_ip, dst_ip, proto, reserved, tcp_len(tcp_header + tcp_payload)
    csum = tcp_v4_check(tcp_len, p_head->ip_saddr, p_head->ip_daddr, csum);
    if (csum != src_check) {
        ret = 1;
        dump("error: tcp calc checksum is 0x%04x, src checksum is 0x%04x\n", ntohs(csum), ntohs(src_check)); 
    }
    else {
        dump_debug("tcp calc checksum is 0x%04x, src checksum is 0x%04x\n", ntohs(csum), ntohs(src_check)); 
    }

    if (update) {
        p_head->tcp_check = csum;
    }
    else {
        p_head->tcp_check = src_check;
    }
    
    return ret;
}

u32 verify_ip_checksum(u8 *p_data, u16 data_len, u8 update)
{
    u32 ret = 0;
    frame_header_no_hdr_extend *p_head = (frame_header_no_hdr_extend*)p_data; 

    //ip_check includes ip_header check only
    u16 src_check = p_head->ip_check;
    p_head->ip_check = 0;
    u32 csum = 0;
    csum = ip_fast_csum((u8*)(&(p_head->ip_hdr)), p_head->ip_hdr_len);
    if (csum != src_check) {
        ret = 1;
        dump("error: ip calc checksum is %04x, src checksum is %04x\n", ntohs(csum), ntohs(src_check)); 
    }
    else {
        dump_debug("ip calc checksum is %04x, src checksum is %04x\n", ntohs(csum), ntohs(src_check)); 
    }

    if (update) {
        p_head->ip_check = csum;
    }
    else {
        p_head->ip_check = src_check;
    }

    return ret;
}

u32 verify_checksum(u8 *p_data, u16 data_len, u8 update)
{
    u32 ret = 0;
    do {
        ret = verify_tcp_checksum(p_data, data_len, update);
        if (0 != ret) {
            break;
        }

        ret = verify_ip_checksum(p_data, data_len, update);
    } while(0);
    return ret;
}

s32 modify_eth_flag(s32 fd, char *p_eth_name, u8 cmd, u32 flag)
{
    s32 ret = 0;

    do {
        struct ifreq ifr;
        bzero(&ifr, sizeof(ifr));
        strncpy(ifr.ifr_name, p_eth_name, sizeof(ifr.ifr_name)-1);
        ret = ioctl(fd, SIOCGIFFLAGS, &ifr);
        if (0 != ret) {
            dump("fail to get if flags\n");
            break;
        }

        dump_debug("modify eth flag, cmd is %d\n", cmd);
        if (OPEN_ETH_FLAG == cmd) {  //open
            ifr.ifr_flags |= flag;
        }
        else {           //close
            ifr.ifr_flags &= ~flag;
        }

        ret = ioctl(fd, SIOCSIFFLAGS, &ifr);
        if (0 != ret) {
            dump("fail to open promisc\n");
            break;
        }
    } while(0);

    return ret;
}


void parse_data(u8 *p_data, u32 data_len, msg_info *p_msg_info)
{
    if ((NULL == p_data) || (NULL == p_msg_info) || (data_len < ETH_MIN_FRAME_LEN) || (data_len > ETH_MAX_FRAME_LEN)) {
        dump("null pointer, data len is %d\n", data_len);
        return;
    }

    frame_header_no_hdr_extend *p_head = (frame_header_no_hdr_extend*)p_data;
    if (IS_BROADCAST(p_head)) {
        p_msg_info->msg_type_id = PROTO_BROADCAST;
        dump_debug("broadcast msg\n");
        return;
    }

    u16 ether_type = ntohs(p_head->ether_hdr_type);
    if (ETHERTYPE_IP == ether_type) {
        if(!IS_NO_IP_EXTEND_HEADER(p_head)) {
            return;
        }

        u8 proto_id = p_head->ip_protocol;
        if (IPPROTO_UDP == proto_id) {
            u16 src_port = ntohs(p_head->udp_src_port);
            if (DHCP_REQ_PORT == src_port) {
                p_msg_info->msg_type_id = PROTO_DHCP_REQ;
            }
            else if (DHCP_RESP_PORT == src_port) { 
                p_msg_info->msg_type_id = PROTO_DHCP_RESP;
            }
            else {
                p_msg_info->msg_type_id = PROTO_UDP_NORMAL;
            }
        }
        else if (IPPROTO_TCP == proto_id) {
            if(IS_NO_IP_EXTEND_HEADER(p_head)) {
                ; //
            }
            p_msg_info->msg_type_id = PROTO_TCP;
        }
        else {
            dump_debug("other ip msg, proto id is %d\n", proto_id);
        }
    } 
    else if (ETHERTYPE_ARP == ether_type) {
        u16 op_cmd = ntohs(p_head->arp_op_cmd);
        if (ARPOP_REQUEST == op_cmd) {
            p_msg_info->msg_type_id = PROTO_ARP_REQ;
        }
        else if (ARPOP_REPLY == op_cmd) {
            p_msg_info->msg_type_id = PROTO_ARP_REPLY;
        }
        else {
            dump_debug("other arp msg, op is 0x%4x\n", op_cmd);
        }
    } 
    else {
        dump_debug("other msg, ether type is 0x%04x\n", ether_type);
    } 
}

u8 is_cfg_dump_proto_msg(msg_info *p_msg_info)
{
    u8 ret = 0;
    if (NULL == p_msg_info) {
        dump("null pointer\n");
        return 0;
    }
    
    u32 cur_proto_id = (p_msg_info->msg_type_id) & proto_mask;
    u32 cur_sub_proto_id = (p_msg_info->msg_type_id) & sub_proto_mask;
    u16 cur_proto_info_index = (p_msg_info->msg_type_id) & proto_info_mask;
   
    dump_debug("cfg prot id is %08x, cfg sub proto id is %08x, cur proto id is %08x, cur sub proto id is %08x\n", 
                                cfg_dump_proto_id, cfg_dump_sub_proto_id, cur_proto_id, cur_sub_proto_id);  
    if ((0 == cfg_dump_proto_id)
        || ((cfg_dump_proto_id == cur_proto_id) 
           && ((0 == cfg_dump_sub_proto_id) || (cfg_dump_sub_proto_id == cur_sub_proto_id)))) {
        dump("msg type is %s\n", msg_infos[cur_proto_info_index]);
        ret = 1;
    }

    return ret;
}

s32 set_socket_packet(u32 domain, u32 type, u32 fd)
{
    if(PF_PACKET != domain) {
        return -1;
    } 

    s32 ret = 0;
    do {
        if (SOCK_PACKET != type) {
            u32 opt_value = 1;
            if (1 == cfg_ignore_outgoing) {
                ret = setsockopt(fd, SOL_PACKET, PACKET_IGNORE_OUTGOING, (u8*)&opt_value, sizeof(opt_value));
                if (0 != ret) {
                    dump("fail to set sock, err is : %s\n", strerror(errno));
                    break;
                }    
            } 

            struct ifreq ifr;
            bzero(&ifr, sizeof(ifr)); 
            strncpy(ifr.ifr_name, cfg_p_eth_name, sizeof(cfg_p_eth_name) - 1);    
            ret = ioctl(fd, SIOCGIFINDEX, &ifr);   //SIOCSIFFLAGS, SIOCGIFADDR, SIOCGIFHWADDR, SIOCGIFTXQLEN, in net/core/dev_ioctl.c
            if (0 != ret) {
                dump("fail to get eth index, err is : %s\n", strerror(errno));
                break;
            }
            
            struct sockaddr_ll addr;
            bzero(&addr, sizeof(addr));    
            addr.sll_family = domain;
            addr.sll_ifindex = ifr.ifr_ifindex;
            ret = bind(fd, (struct sockaddr*)&addr, sizeof(addr));
            if (0 != ret) {
                dump("fail to bind\n");
                break;
            }
        }
        else {
            struct sockaddr addr;
            bzero(&addr, sizeof(addr));
            addr.sa_family = domain;
            strncpy(addr.sa_data, cfg_p_eth_name, sizeof(addr.sa_data) - 1);
            ret = bind(fd, &addr, sizeof(addr));
            if (0 != ret) {
                dump("fail to bind\n");
                break;
            }
        }
    } while (0);

    return ret;
}

u32 get_ipv4(u32 fd)
{
    u32 ip = 0;
    struct ifreq ifr;
    bzero(&ifr, sizeof(ifr)); 
    strncpy(ifr.ifr_name, cfg_p_eth_name, sizeof(cfg_p_eth_name) - 1);    
    s32 ret = ioctl(fd, SIOCGIFADDR, &ifr);   //SIOCSIFFLAGS, SIOCGIFADDR, SIOCGIFHWADDR, SIOCGIFTXQLEN, in net/core/dev_ioctl.c
    if (0 == ret) {
        ip = ntohl((*((struct sockaddr_in*)(&ifr.ifr_addr))).sin_addr.s_addr);
    }
    else {
        dump("fail to get eth ip, err is : %s\n", strerror(errno));
    }

    return ip;
}

void *recv_thread(void *p_args)
{
    s32 ret = 0;
    s32 fd = 0;
    s32 epfd = 0;
    
    do {
        s32 domain = PF_PACKET;
        //s32 type = SOCK_PACKET;
        s32 type = SOCK_RAW;
        s32 protocol = htons((u16)(ETH_P_ALL));
        fd = socket(domain, type, protocol);    
        if (-1 == fd) {
            dump("fail to create socket\n");
            ret = fd;
            break;
        }

        if (PF_PACKET == domain) {
            ret = set_socket_packet(domain, type, fd);
            if (0 != ret) {
                    break;
            }
        }

        u32 ip = get_ipv4(fd);
        if (0 == ip) {
            dump("ip is %d\n", ip);
            break;
        } 

        eth_ip = ip;
        ret = modify_eth_flag(fd, cfg_p_eth_name, OPEN_ETH_FLAG, IFF_PROMISC);
        if (0 != ret) {
            dump("fail to open promisc\n");
            break;
        }

        s32 max_events = 20;
        epfd = epoll_create(max_events);
        if (-1 == epfd) {
            dump("fail to create epoll\n");
            break;
        }

        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = fd;
        if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev)) {
            dump("fail to add %d to epoll\n", fd);
            break;
        }

        struct epoll_event events[max_events];
        memset(events, 0, sizeof(events));
        struct sockaddr remote_addr;
        bzero(&remote_addr, sizeof(remote_addr));
        strncpy(remote_addr.sa_data, cfg_p_eth_name, sizeof(remote_addr.sa_data) - 1);
        socklen_t remote_addr_len = sizeof(remote_addr);

        /* ETH_FRAME_LEN = 1514; ETH_ZLEN = 60; ETH_CRC_LEN = 4; 
           ETHER_MAX_LEN=ETH_FRAME_LEN + ETHER_CRC_LEN, ETHER_MIN_LEN=ETH_ZLEN + ETHER_CRC_LEN, defined in net/ethernet.h*/
        u32 max_data_size = ETH_FRAME_LEN;  
        u8 buf[max_data_size] = {0};
        u32 dump_count = 0;
        u32 cur_proto_id = 0;
        u32 cur_sub_proto_id = 0;
        u16 cur_proto_info_index = 0;
        msg_info msg_info_tmp;
        s32 len = 0;

        while (recv_thread_running) {
            s32 fds = epoll_wait(epfd, events, max_events, 200);
            u32 i = 0;
            for (i = 0; i < fds; i++) {
                if ((events[i].events & EPOLLIN) || (events[i].events & EPOLLPRI)) {
                    s32 fd_tmp = events[i].data.fd; 
                    if (fd_tmp < 0) {
                        dump("recv fd_tmp < 0\n");
                        continue;
                    }

                    if (fd != fd_tmp) {
                        continue;
                    }
                            
                    memset(buf, 0, max_data_size);
                    len = recvfrom(fd_tmp, buf, max_data_size, 0, NULL, NULL);  //send: sendto(fd_tmp, buf, len, MSG_DONOTWAIT, NULL, 0);
                    if ((len < ETH_MIN_FRAME_LEN) || (len > ETH_MAX_FRAME_LEN)) {
                        dump("len is wrong : %d\n", len);
                    }

                    memset(&msg_info_tmp, 0, sizeof(msg_info_tmp));
                    parse_data(buf, len, &msg_info_tmp);
               
                    if (PROTO_TCP == msg_info_tmp.msg_type_id) {
                        ret = verify_checksum(buf, len, 0);
                        if (ret != 0) {
                            //continue;
                        }
                    }

                    if ((1 == is_cfg_dump_proto_msg(&msg_info_tmp)) && (dump_count < cfg_dump_count)) {    
                        dump_data(buf, len);
                        dump_count++;
                    }
                }
            }

            if (dump_count >= cfg_dump_count) {
                break;
            }
        }
    } while(0);

    ret = modify_eth_flag(fd, cfg_p_eth_name, CLOSE_ETH_FLAG, IFF_PROMISC);
    if (0 != ret) {
        dump("fail to close promisc\n");
    }

    if (fd >= 0) {
        close(fd);
    }

    if (epfd >= 0) {
        close(epfd);
    }

    return ((void*)(0));
}


void signal_handler( int signum )
{
    printf("Interrupt signal %d\n", signum);
    recv_thread_running = 0; 
}

void help()
{
    dump("for example: ./packet_dump -t -c 2\n");
}

s32 parse_args(s32 argc, char *argv[])
{
    s32 ret = 0;
    char *short_opts = (char*)"i:abudtc:s::hRCN";
    struct option long_opts[] = {
        {"eth",        required_argument, NULL, 'i'},
        {"arp",        no_argument,       NULL, 'a'},
        {"broadcast",  no_argument,       NULL, 'b'},
        {"udp",        no_argument,       NULL, 'u'},
        {"dhcp",       no_argument,       NULL, 'd'},
        {"tcp",        no_argument,       NULL, 't'},
        {"count",      required_argument, NULL, 'c'},
        {"dump_len",   optional_argument, NULL, 's'},
        {"recv",       optional_argument, NULL, 'R'},
        {"help",       no_argument,       NULL, 'h'},
        {"checksum",   no_argument,       NULL, 'C'},
        {"new_packet", no_argument,       NULL, 'N'},
        {0, 0, 0, 0},
    };

    int opt = 0;;
    int opt_index = 0;
    while((opt = getopt_long(argc, argv, short_opts, long_opts, &opt_index)) != -1){
        switch(opt) {
            case 'i':
                dump_debug("%s\n", optarg);
                cfg_p_eth_name = optarg;          // optarg is argv[optind-1]
                break;
            case 'a':
                cfg_dump_proto = PROTO_ARP;
                break;
            case 'b':
                dump_debug("%s\n", argv[optind-1]);
                cfg_dump_proto = PROTO_BROADCAST;
                break;
            case 'C':
                test_for_checksum();
                ret = 1;
                return ret;
            case 'N':
                test_for_create_new_tcp_packet();
                ret = 1;
                return ret;
            case 'u':
                dump_debug("%s\n", argv[optind-1]);
                cfg_dump_proto = PROTO_UDP;
                break;
            case 'd':
                dump_debug("%s\n", argv[optind-1]);
                cfg_dump_proto = PROTO_DHCP;
                break;
            case 't':
                dump_debug("%s\n", argv[optind-1]);
                cfg_dump_proto = PROTO_TCP;
                break;
            case 'c':
                dump_debug("%s\n", argv[optind-1]);
                cfg_dump_count = atoi(argv[optind - 1]);
                break;
            case 's':
                dump_debug("%s\n", argv[optind - 1]);
                dump_debug("%s\n", argv[optind]);
                cfg_dump_len = atoi(argv[optind]);
                break;
            case 'h':
                help();
                exit(0);
            case 'R':
                cfg_ignore_outgoing = 1;
                break;
            default:
                help();
                ret = -1;
                return ret;
        }
    }

    if (optind > argc) { 
        dump("wrong optind %d\n", optind);
        ret = -1;
    }
    else {
        dump_debug("cfg proto is %08x\n", cfg_dump_proto);
        cfg_dump_proto_id = cfg_dump_proto & proto_mask;
        cfg_dump_sub_proto_id = cfg_dump_proto & sub_proto_mask;
    }

    return ret;
}

#include "packet.data"
void test_for_create_new_tcp_packet()
{ 
    u32 data_len = sizeof(packet_data); 

    msg_info msg_info_tmp;
    memset(&msg_info_tmp, 0, sizeof(msg_info_tmp));
    parse_data(packet_data, data_len, &msg_info_tmp);       
    if (PROTO_TCP != msg_info_tmp.msg_type_id) {
        dump("no tcp data needed\n");
        return;
    }

    frame_header_no_hdr_extend *p_old_head = (frame_header_no_hdr_extend*)packet_data;
    u8 data[255] = {0};
    memcpy(data, packet_data, data_len);
    u8 *p_data = &data[0]; 
    u8 ip_header_offset = 14;
    u8 iphdr_hlen = 20;
    u16 tcp_header_offset = ip_header_offset + iphdr_hlen;
    struct tcphdr *p_tcphdr = (struct tcphdr*)(&p_data[tcp_header_offset]);

    /* change_port */
    p_tcphdr->source = p_old_head->tcp_dst_port;
    p_tcphdr->dest = p_old_head->tcp_src_port;

    /* change sequence num */
    p_tcphdr->seq = p_old_head->tcp_ack_seq;
    u16 old_payload_len = ntohs(p_old_head->ip_tot_len) - (p_old_head->ip_hdr_len * 4) - (p_old_head->tcp_hdr_len * 4);
    dump("payload len is %d\n", old_payload_len);
    p_tcphdr->ack_seq = htonl(ntohl(p_old_head->tcp_seq) + old_payload_len);

    /* change src addr and dst addr */
    struct iphdr *p_iphdr = (struct iphdr*)(&p_data[ip_header_offset]);
    p_iphdr->saddr = p_old_head->ip_daddr;
    p_iphdr->daddr = p_old_head->ip_saddr;

    /* checksum */
    verify_checksum(p_data, data_len, 1);

    /* change mac */
    struct ether_header *p_ether_hdr = (struct ether_header*)p_data;
    memcpy(p_ether_hdr->ether_dhost, p_old_head->p_ether_hdr_src_mac, ETHER_ADDR_LEN);
    memcpy(p_ether_hdr->ether_shost, p_old_head->p_ether_hdr_dst_mac, ETHER_ADDR_LEN);
    dump_data(p_data, data_len);
}

void test_for_checksum() 
{
    verify_checksum(packet_data, sizeof(packet_data), 0);
}  


s32 main(s32 argc, s8 *argv[])
{
    pthread_t ptd = 0;
    void *p_args = NULL;
    s32 ret = 0;
   
    do {
        ret = parse_args(argc, argv);
        if (ret < 0) {
            dump("fail to parse args\n");
            break;
        }
        else if (ret > 0) {
            ret = 0;
            break;
        }

        int signum = SIGINT; //SIGTERM;//SIGINT;
        sighandler_t sighandler = signal(signum, signal_handler);
        if (SIG_ERR == sighandler) {
            dump("fail to signal\n");
            ret = -1;
            break;
        }

        ret = pthread_create(&ptd, NULL, recv_thread, p_args);
        if (0 != ret) {
            dump("fail to create thread\n");
            break;
        }

        pthread_setname_np(ptd, "recv_packet_thread"); 
    } while (0);

    if (0 != ptd) {
        pthread_join(ptd, NULL);
        ptd = 0;
    }

    return ret;
}
