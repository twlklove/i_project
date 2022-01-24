#ifndef __I_ETH_H__
#define __I_ETH_H__
#include "types.h"

typedef struct {
   char mac[6];
}mac_addr;

typedef struct {
   mac_addr dst_mac;
   mac_addr src_mac;
}dst_src_mac_addr;

typedef struct {
    union {
        dst_src_mac_addr mac;
        struct {
            int mac_0;
            int mac_0_1;
            int mac_1;
        }__mac_addr;
    }_mac_addr;
    short type;

    #define mac   _mac_addr.mac
    #define mac_0 _mac_addr.__mac_addr.mac_0
    #define mac_0_1 _mac_addr.__mac_addr.mac_0_1
    #define mac_1 _mac_addr.__mac_addr.mac_1
}mac_1eader;
//__attribute__((aligned(4))) mac_1eader;
//__attribute__ ((packed)) mac_1eader; no aligned
//

#endif
