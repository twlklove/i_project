#include <iostream>
#include <string> 

using namespace std;

class Test
{
private:
    int value;

public:
    Test()
	{
	    value = 100;
	}

    void dump()
	{
	    cout << value << endl;
	}
};

class Singleton : Test
{
private:
	Singleton(int val)
	{
	    this->value = val;
	}

    int value;
	static Singleton* p_instance;

public:
	static Singleton* getInstance(int val)
	{
	    if (NULL == p_instance) {
	        p_instance = new Singleton(val);
	    }
	    
	    return p_instance;
    }

	void dump()
	{
	    cout << value << endl;
	}
};

Singleton * Singleton::p_instance = NULL;

int main()
{

    Test test;
	test.dump();

    Singleton *sig = Singleton::getInstance(12);
	sig->dump();

	Singleton *sig1 = Singleton::getInstance(13);
	sig->dump();
	sig1->dump();


	return 0;
}
