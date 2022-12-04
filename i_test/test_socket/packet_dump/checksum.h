/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * INET		An implementation of the TCP/IP protocol suite for the LINUX
 *		operating system.  INET is implemented using the  BSD Socket
 *		interface as the means of communication with the user level.
 *
 *		Checksumming functions for IP, TCP, UDP and so on
 *
 * Authors:	Jorge Cwik, <jorge@laser.satlink.net>
 *		Arnt Gulbrandsen, <agulbra@nvg.unit.no>
 *		Borrows very liberally from tcp.c and ip.c, see those
 *		files for more names.
 */

// as is in include/net/checksum.h

#ifndef _CHECKSUM_H
#define _CHECKSUM_H

#include "types.h"

#define __force

/* add by me*/
typedef u16  __sum16;
typedef u32  __wsum;

/* for ip checksum */

static inline unsigned short from32to16(unsigned int x);
static unsigned int do_csum(const unsigned char *buff, int len);
__sum16 ip_fast_csum(const void *iph, unsigned int ihl);

__wsum csum_partial(const void *buff, int len, __wsum wsum);


/* for tcp checksum  */
// as is in linux/include/asm-generic/checksum.h
static inline u32 from64to32(u64 x);
__wsum csum_tcpudp_nofold(u32 saddr, u32 daddr, u32 len, u8 proto, __wsum sum);
 
#ifndef csum_fold
/*
 * Fold a partial checksum
 */
static inline __sum16 csum_fold(__wsum csum)
{
	u32 sum = (__force u32)csum;
	sum = (sum & 0xffff) + (sum >> 16);
	sum = (sum & 0xffff) + (sum >> 16);
	return (__force __sum16)~sum;
}
#endif

#ifndef csum_tcpudp_magic
static inline __sum16
csum_tcpudp_magic(u32 saddr, u32 daddr, u32 len, u8 proto, __wsum sum)
{
	return csum_fold(csum_tcpudp_nofold(saddr, daddr, len, proto, sum));
}
#endif

#endif

