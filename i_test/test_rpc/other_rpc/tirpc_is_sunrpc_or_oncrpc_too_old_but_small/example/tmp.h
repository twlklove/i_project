/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#ifndef _TMP_H_RPCGEN
#define _TMP_H_RPCGEN

#include <rpc/rpc.h>


#ifdef __cplusplus
extern "C" {
#endif

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

struct TEST {
	int op;
	float arg1;
	float arg2;
	float result;
};
typedef struct TEST TEST;

#define TEST_PROG 0x20000001
#define TEST_VER 2

#if defined(__STDC__) || defined(__cplusplus)
#define TEST_PROC 1
extern  struct TEST * test_proc_2(struct TEST *, CLIENT *);
extern  struct TEST * test_proc_2_svc(struct TEST *, struct svc_req *);
extern int test_prog_2_freeresult (SVCXPRT *, xdrproc_t, caddr_t);

#else /* K&R C */
#define TEST_PROC 1
extern  struct TEST * test_proc_2();
extern  struct TEST * test_proc_2_svc();
extern int test_prog_2_freeresult ();
#endif /* K&R C */

/* the xdr functions */

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t xdr_TEST (XDR *, TEST*);

#else /* K&R C */
extern bool_t xdr_TEST ();

#endif /* K&R C */

#ifdef __cplusplus
}
#endif

#endif /* !_TMP_H_RPCGEN */
