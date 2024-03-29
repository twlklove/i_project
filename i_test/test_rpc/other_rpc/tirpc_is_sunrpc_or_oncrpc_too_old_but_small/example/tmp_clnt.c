/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#include <memory.h> /* for memset */
#include "tmp.h"

/* Default timeout can be changed using clnt_control() */
static struct timeval TIMEOUT = { 25, 0 };

struct TEST *
test_proc_2(struct TEST *argp, CLIENT *clnt)
{
	static struct TEST clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call (clnt, TEST_PROC,
		(xdrproc_t) xdr_TEST, (caddr_t) argp,
		(xdrproc_t) xdr_TEST, (caddr_t) &clnt_res,
		TIMEOUT) != RPC_SUCCESS) {
		return (NULL);
	}
	return (&clnt_res);
}
