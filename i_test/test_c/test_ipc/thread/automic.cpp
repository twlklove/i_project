#include <atomic>
#include <thread>
#include <iostream>
#include <thread>
using namespace std;

//atomic<T>类模板，如atomic_int64_t是通过typedef atomic<int64_t> atomic_int64_t
atomic_int64_t total(0); 

void threadFunc(int64_t endNum)
{
	for (int64_t i = 1; i <= endNum; ++i)
	{
		total += i;
	}
}

void threadFunc1(int64_t endNum)
{
	for (int64_t i = 1; i <= endNum; ++i)
	{
		total -= i;
	}
}

int main()
{
	int64_t endNum = 100;
	thread t1(threadFunc, endNum);
	thread t2(threadFunc1, endNum);

	t1.join();
	t2.join();

	cout << "total=" << total << endl; //10100
}
