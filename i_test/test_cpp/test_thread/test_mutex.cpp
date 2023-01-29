#include <iostream>       // cout
#include <thread>         // thread, this_thread::sleep_for
#include <chrono>         // chrono::seconds
#include <mutex>          // mutex, timed_mutex, unique_lock
#include <ctime>          // time_t, tm, localtime, mktime

using namespace std;

volatile int counter (0); // non-atomic counter
mutex mtx;           // locks access to counter

void pause_thread(int n) 
{
    this_thread::sleep_for(chrono::seconds(n));
    thread::id thread_id = this_thread::get_id();
    cout << "thread " << thread_id  << " pause of " << n << " seconds ended\n";

    for (int i=0; i<100; ++i) {
        if (mtx.try_lock()) {   // only increase if currently not locked:
          ++counter;
          mtx.unlock();
        }
    }
}

void test_0() 
{
  thread threads[5];                         // default-constructed threads

  cout << "Spawning 5 threads...\n";
  for (int i=0; i<5; ++i) {
    threads[i] = thread(pause_thread,i+1);   // move-assign threads
    if (2 == i) {
        threads[i].detach();    
    }
  }

  cout << "Done spawning threads. Now waiting for them to join:\n";
  for (int i=0; i<5; ++i)
    if (threads[i].joinable()) {
        cout << "thread " << i << " join" << endl;
        threads[i].join();
    }

  cout << "All threads joined!\n";
  cout << counter << " successful increases of the counter.\n";
}

timed_mutex cinderella;

// gets time_point for next midnight:
chrono::time_point<chrono::system_clock> midnight() {
  using chrono::system_clock;
  time_t tt = system_clock::to_time_t (system_clock::now());
  struct tm * ptm = localtime(&tt);
  ++ptm->tm_mday; ptm->tm_hour=0; ptm->tm_min=0; ptm->tm_sec=0;
  return system_clock::from_time_t (mktime(ptm));
}

void carriage() {
  if (cinderella.try_lock_until(midnight())) {        // try_lock_until
    cout << "ride back home on carriage\n";
    cinderella.unlock();
  }
  else
    cout << "carriage reverts to pumpkin\n";
}

void ball() {
  cinderella.lock();
  cout << "at the ball...\n";
  cinderella.unlock();
}

void test_1()
{
  thread th1 (ball);
  thread th2 (carriage);

  th1.join();
  th2.join();
}

timed_mutex timed_mtx;
void fireworks () {
  // waiting to get a lock: each thread prints "-" every 200ms:
  while (!timed_mtx.try_lock_for(chrono::milliseconds(200))) {     // try_lock_for
    cout << "-";
  }

  // got a lock! - wait for 1s, then this thread prints "*"
  this_thread::sleep_for(chrono::milliseconds(1000));
  cout << "*\n";
  timed_mtx.unlock();
}

void test_2()
{
  thread threads[10];
  // spawn 10 threads:
  for (int i=0; i<10; ++i)
    threads[i] = thread(fireworks);

  for (auto& th : threads) th.join();
}

void print_block (int n, char c) {
  // critical section (exclusive access to cout signaled by lifetime of lck):
  unique_lock<mutex> lck(mtx);                               // unique_lock
  for (int i=0; i<n; ++i) { cout << c; }
  cout << '\n';
}

void test_3()
{
  thread th1 (print_block,550,'*');
  thread th2 (print_block,550,'$');

  th1.join();
  th2.join();
}

int main() 
{
    test_0();
    test_1();
    test_2();
    test_3();

    return 0;
}


