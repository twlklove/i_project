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
#include <net/if.h>

#include <iostream>
#include <cstring>
#include <vector>
#include <list>
#include <set>
#include <map>

typedef unsigned char          u8; 
typedef unsigned short int     u16;
typedef unsigned int           u32;
typedef unsigned long long     u64;

typedef char                   s8;
typedef short int              s16;
typedef int                    s32;
typedef long long              s64;


#define ETH   ((s8*)("ens33"))

#define DHCP_REQ_PORT     0x44
#define DHCP_RESP_PORT    0x43

/* etherner_packet = 14B ether_hdr + protocol_data + 4B CRC */
#define ETH_MIN_FRAME_LEN  42           //   42B: 14B ether_hdr_len + 28Barp_hdr_len
#define ETH_MAX_FRAME_LEN  ETH_FRAME_LEN  // 1514B: 14B ether_hdr_len + 1500B MTU

#ifndef dump
#define dump(a, ...)                printf("%s[%d] mark: " a, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define DEFAULT_DUMP_LEN   80
#define DEBUG 0

#if DEBUG
#undef  dump_debug 
#define dump_debug(a, ...)      dump(a, ##__VA_ARGS__)
#else
#define dump_debug(a, ...) 
#endif

#define output_data             printf
#define dump_data(p_data, len)                                             \
({                                                                         \
    s32 i =0;                                                              \
    output_data("len is %d", len);                                         \
    len = len > cfg_dump_len ? cfg_dump_len : len;                         \
    for(i = 0; i < len; i++) {                                             \
        if (0 == i % 16) {                                                 \
            output_data("\n");                                             \
	    output_data("0x%04x: ", i);                                    \
        }                                                                  \
                                                                           \
        if (i + 1 < len) {                                                 \
            output_data("%02x%02x ", p_data[i]&0xFF, p_data[i+1]&0xFF);    \
            i++;                                                           \
        }                                                                  \
        else {                                                             \
            output_data("%02x", p_data[i]&0xFF);                           \
        }                                                                  \
    }                                                                      \
    output_data("\n\n");                                                   \
})

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
const char *msg_infos[] = {"pacekt", "arp", "arp req", "arp reply", "udp", "broadcast", 
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

#define p_ether_hdr_dst_mac          ether_hdr.ether_dhost
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
        u64 dst_mac_0 = ntohl(*((u32*)(p_head->p_ether_hdr_dst_mac)));         \
        u64 dst_mac_1 = ntohs(*((u16*)(p_head->p_ether_hdr_dst_mac + 4)));     \
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
	if (IP_HEADER_BASE_SIZE == p_head->ip_hdr_len * 4) {                   \
            ret = 1;                                                           \
        }                                                                      \
    } while(0);                                                                \
    ret;                                                                       \
})

#define IS_NO_TCP_EXTEND_HEADER(p_head)                                        \
({                                                                             \
    s32 ret = 0;                                                               \
    do {                                                                       \
	if (TCP_HEADER_BASE_SIZE == p_head->tcp_hdr_len * 4) {                 \
            ret = 1;                                                           \
        }                                                                      \
    } while(0);                                                                \
    ret;                                                                       \
})

}__attribute__((packed)) frame_header_no_hdr_extend;
//#paragma pack()

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
    if ((NULL == p_data) || (NULL == p_msg_info) || (data_len < ETH_MIN_FRAME_LEN) || (data_len > ETH_MAX_FRAME_LEN)){
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

u32 recv_thread_running = 1;
s8 *cfg_p_eth_name = ETH;
msg_type cfg_proto = PROTO_PACKET;
u32 cfg_dump_count = 0xFFFFFFFF;
u32 cfg_dump_len = DEFAULT_DUMP_LEN;
const u32 proto_mask = 0xFF000000;
const u32 sub_proto_mask = 0x00FF0000;
const u32 proto_info_mask  = 0x0000FFFF;
u32 cfg_proto_id = 0; 
u32 cfg_sub_proto_id = 0;

void *recv_thread(void *p_args)
{
    s32 ret = 0;
    s32 fd = 0;
    s32 epfd = 0;
    
    do {
        s32 domain = PF_PACKET;
        s32 protocol = htons((u16)(ETH_P_ALL));
        fd = socket(domain, SOCK_PACKET, protocol);
        if (-1 == fd) {
            dump("fail to create socket\n");
            ret = fd;
            break;
        }

	
        ret = modify_eth_flag(fd, cfg_p_eth_name, OPEN_ETH_FLAG, IFF_PROMISC);
	if (0 != ret) {
            dump("fail to open promisc\n");
	    break;
	}

	struct sockaddr addr;
	bzero(&addr, sizeof(addr));
	addr.sa_family = PF_PACKET;
	strncpy(addr.sa_data, cfg_p_eth_name, sizeof(addr.sa_data)-1);
	ret = bind(fd, &addr, sizeof(addr));
	if (0 != ret) {
	    dump("fail to bind\n");
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

	    	    s32 len = recvfrom(fd_tmp, buf, max_data_size, 0, &remote_addr, &remote_addr_len);
		    if ((len < ETH_MIN_FRAME_LEN) || (len > ETH_MAX_FRAME_LEN)) {
			dump("len is wrong : %d\n", len);
		    }
 
		    memset(&msg_info_tmp, 0, sizeof(msg_info_tmp));
		    parse_data(buf, len, &msg_info_tmp);

		    cur_proto_id = (msg_info_tmp.msg_type_id) & proto_mask;
		    cur_sub_proto_id = (msg_info_tmp.msg_type_id) & sub_proto_mask;
		    cur_proto_info_index = (msg_info_tmp.msg_type_id) & proto_info_mask;
		     
		    if ((0 == cfg_proto_id)
		        || ((cfg_proto_id == cur_proto_id) 
		           && ((0 == cfg_sub_proto_id) || (cfg_sub_proto_id == cur_sub_proto_id)))) {
			if(dump_count++ < cfg_dump_count){
			    dump("msg type is %s\n", msg_infos[cur_proto_info_index]);
			    dump_data(buf, len);
			}
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
    dump("help\n");
}

s32 parse_args(s32 argc, char *argv[])
{
    s32 ret = 0;
    char *short_opts = (char*)"i:abudtc:s::h";
    struct option long_opts[] = {
	{"eth",        required_argument, NULL, 'i'},
        {"arp",        no_argument,       NULL, 'a'},
	{"broadcast",  no_argument,       NULL, 'b'},
        {"udp",        no_argument,       NULL, 'u'},
        {"dhcp",       no_argument,       NULL, 'd'},
        {"tcp",        no_argument,       NULL, 't'},
	{"count",      required_argument, NULL, 'c'},
	{"dump_len",   optional_argument, NULL, 's'},
	{"help",       no_argument,       NULL, 'h'},
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
		dump_debug("%s\n", argv[optind-1]);
                cfg_proto = PROTO_ARP;
                break;
            case 'b':
		dump_debug("%s\n", argv[optind-1]);
                cfg_proto = PROTO_BROADCAST;
                break;
            case 'u':
		dump_debug("%s\n", argv[optind-1]);
                cfg_proto = PROTO_UDP;
                break;
	    case 'd':
		dump_debug("%s\n", argv[optind-1]);
		cfg_proto = PROTO_DHCP;
                break;
	    case 't':
		dump_debug("%s\n", argv[optind-1]);
                cfg_proto = PROTO_TCP;
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
        cfg_proto_id = cfg_proto & proto_mask;
        cfg_sub_proto_id == cfg_proto & sub_proto_mask;
    }

    return ret;
}

s32 main(s32 argc, s8 *argv[])
{
    pthread_t ptd = 0;
    void *p_args = NULL;
    s32 ret = 0;

    do {
	ret = parse_args(argc, argv);
        if (0 != ret) {
            dump("fail to parse args\n");
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
