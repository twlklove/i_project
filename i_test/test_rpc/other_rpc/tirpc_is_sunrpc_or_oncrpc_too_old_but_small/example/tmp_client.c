/*
 * This is sample code generated by rpcgen.
 * These are only templates and you can use them
 * as a guideline for developing your own functions.
 */

#include "tmp.h"


void
test_prog_2(char *host)
{
	CLIENT *clnt;
	struct TEST  *result_1;
	struct TEST  test_proc_2_arg;

	/* added by myself */
	printf("please input operation\n\t0---ADD\n\t1--SUB\n");
	char c = getchar();
	switch(c) {
		case '0' :
			test_proc_2_arg.op = ADD;
			break;
		case '1' :
			test_proc_2_arg.op = SUB;
			break;
		default :
			printf("operation wrong\n");
			break;
	}

	printf("please input the first num\n");
	scanf("%f", &test_proc_2_arg.arg1); 

	printf("please input the second num\n");
	scanf("%f", &test_proc_2_arg.arg2); 
	/* added by myself */


#ifndef	DEBUG
	clnt = clnt_create (host, TEST_PROG, TEST_VER, "udp");
	if (clnt == NULL) {
		clnt_pcreateerror (host);
		exit (1);
	}
#endif	/* DEBUG */

	result_1 = test_proc_2(&test_proc_2_arg, clnt);
	if (result_1 == (struct TEST *) NULL) {
		clnt_perror (clnt, "call failed");
	}
#ifndef	DEBUG
	clnt_destroy (clnt);
#endif	 /* DEBUG */

	/* added by myself */
	printf("result is %0.3f\n", result_1->result);
}


int
main (int argc, char *argv[])
{
	char *host;

	if (argc < 2) {
		printf ("usage: %s server_host\n", argv[0]);
		exit (1);
	}
	host = argv[1];
	test_prog_2 (host);
exit (0);
}
