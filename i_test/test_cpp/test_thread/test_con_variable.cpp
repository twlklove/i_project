// condition_variable example
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id (int id) {
  std::unique_lock<std::mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  std::cout << "thread " << id << '\n';
}

void go() {
  std::unique_lock<std::mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}

void test_0()
{
  std::thread threads[10];
  // spawn 10 threads:
  for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_id,i);

  std::cout << "10 threads ready to race...\n";
  go();                       // go!

  for (auto& th : threads) th.join();
}

int value;

void read_value() {
  std::cin >> value;
  cv.notify_one();
}

void test_1()
{
  std::cout << "Please, enter an integer (I'll be printing dots): \n";
  std::thread th (read_value);

  std::mutex mtx;
  std::unique_lock<std::mutex> lck(mtx);
  while (cv.wait_for(lck,std::chrono::seconds(1))==std::cv_status::timeout) { //if ( wait_until(lck,abs_time) == cv_status::timeout)
    std::cout << '.' << std::endl;
  }
  std::cout << "You entered: " << value << '\n';

  th.join();
}

int main()
{
    test_0();
    test_1();
}
