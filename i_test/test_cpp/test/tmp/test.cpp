#include "test.h"
#include <iostream>
using namespace std;

A::A()
{
     a = 0; 
     b = 0;
}

void A::dump_1()
{
    cout << a << endl;
}

void A::dump_2()
{
    cout << b << endl;
}

A::~A() 
{
    ;
}

B::B()
{
    a = 0;
    b = 0;
}

void B::dump_1()
{
    cout << a + 1 << endl;
}

void B::dump_2()
{
    cout << b + 1 << endl;
}

B::~B() 
{
    ;
}
