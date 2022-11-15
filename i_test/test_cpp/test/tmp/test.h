
class A 
{
	protected:
	       	int a;
		int b;
	public :
		A();
		void dump_1();
		virtual void dump_2();
		virtual ~A();
};

class B: public A
{
	public :
		B();
		void dump_1();
		virtual void dump_2();
		virtual ~B();
};
