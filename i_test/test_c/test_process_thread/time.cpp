#include <ctime>
#include <iostream>
#include <unistd.h>

#include <chrono>
using namespace std::chrono;

using namespace std;

void test_2()
{
    //steady_clock，稳定时钟，保证每时每刻以稳定速率增长 
    // template <class Clock, class Duration = typename Clock::duration> class time_point; //system_clock steady_clock high_resolution_clock
    time_point<high_resolution_clock>_start;
    _start = high_resolution_clock::now();

    sleep(1); 
	cout << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << endl;
    cout << duration_cast<milliseconds>(high_resolution_clock::now() - _start).count() << endl;
    cout << duration_cast<microseconds>(high_resolution_clock::now() - _start).count() << endl;
	cout << duration_cast<nanoseconds>(high_resolution_clock::now() - _start).count() << endl;

}

void test_1()
{
    time_t now = time(0);
	sleep(1);
	time_t now_1 = time(0);
    double diff = difftime(now_1, now);
	cout << now << " " << now_1 <<" " << diff << endl;
}

int main() 
{
	
    test_1();
	test_2();
	return 0;

}
