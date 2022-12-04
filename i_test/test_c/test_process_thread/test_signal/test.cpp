#include <iostream>
#include <csignal>
#include <unistd.h>
 
using namespace std;
 
void signalHandler( int signum )
{
    cout << "Interrupt signal (" << signum << ") received.\n";
 
    // 清理并关闭
    // 终止程序 
 
   exit(signum);  
 
}

int signum = SIGTERM;//SIGINT;
int main ()
{
    int i = 0;
    signal(signum, signalHandler);  
 
    while(++i){
       cout << "Going to sleep...." << endl;
       if( i == 3 ){
          raise(signum);
       }
       sleep(1);
    }
 
    return 0;
}
