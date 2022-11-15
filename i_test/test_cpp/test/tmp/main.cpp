#include "test.h"

int main(int argc, char*argv[])
{
    A a;
    B b;
    A *p = &a;
    p->dump_1();
    p->dump_2();

    p = &b;
    p->dump_1();
    p->dump_2();

    return 0;
}
