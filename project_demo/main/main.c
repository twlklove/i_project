#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"
#include "log.h"
#include "define.h"
#include "i_eth.h"

void test_random()
{
   DUMP(info, "%d\n", atoi("093"));

   time_t t;
   srand((unsigned int)time(&t));
   int value = rand();
   DUMP(info, "%04x\n", value);
   DUMP(info, "%04x\n", value & 0xff);
}

void test_mem_leak()
{
   u32 *p = (u32*)malloc(sizeof(int));
   u32 *p1 = (u32 *)new(sizeof(int));
   *p1 = 20;
   *p = 10;
   DUMP(info, "%u, %u\n", (u32)sizeof(p), (u32)sizeof(*p));
}

void test_eth()
{
    u8 data[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x06, 0x09, 0x0a};
    mac_1eader *recv = (mac_1eader*)data;
    DUMP(info, "%u, %u, %u\n", (u32)sizeof(mac_addr), (u32)sizeof(dst_src_mac_addr), (u32)sizeof(mac_1eader));
    DUMP(info, "%08x, %08x, %08x, %04x\n", recv->mac_0, recv->mac_0_1, recv->mac_1, recv->type);

    u32 eth_mac_0 = B2L_32(recv->mac_0);
    u32 eth_mac_0_1 = B2L_32(recv->mac_0_1);
    u32 eth_mac_1 = B2L_32(recv->mac_1);
    u16 eth_type = B2L_16(recv->type);
 
    DUMP(info, "%08x%04x, %04x%08x, %04x\n", eth_mac_0, (eth_mac_0_1 >> 16)&0xFFFF, (eth_mac_0_1)&0xFFFF, eth_mac_1, eth_type);
}

int main()
{ 
    init();
    test_random();
    test_mem_leak();
    test_eth();
 
    uninit();
    return 0;
}
