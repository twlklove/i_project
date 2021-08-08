#include "first.pb.h"
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

int main(void) 
{ 
	lm::helloworld msg; 
	msg.set_id(101); 
	msg.set_str("hello");

	fstream input("./log", ios::in | ios::binary);
	if (!msg.ParseFromIstream(&input)) {
		cout << "Failed to parse address book." << endl;
		return -1;
	}

	cout << msg.id() << endl;
	cout << msg.str() << endl;

	return 0;
}
