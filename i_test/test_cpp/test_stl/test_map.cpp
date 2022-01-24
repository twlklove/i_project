#include <map>
#include <queue>
#include <iostream>
using namespace std;

int main()
{
    multimap<int, string> data;
    deque<int> q_data;
    int key = 10;
    q_data.front();
    q_data.clear();
    q_data.pop_front();
    q_data.pop_front();
    //deque<int>::iterator iter = q_data.begin();


    data.insert(make_pair(key, "hello"));
    multimap<int ,string>::iterator it = data.find(key);
    if (it == data.end()) {
        cout << "not find key" << endl;
    }
    else {
        cout <<"key is " << it->first <<", value is " << it->second << endl;
    }

    cout << "end" << endl;
    return 0;
}

