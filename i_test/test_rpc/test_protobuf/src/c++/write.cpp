#include <iostream>
#include <fstream>
#include <string>
#include <string>
#include <fstream>
#include <iostream>

#include "hello.pb.h"
using namespace std;
int main(void) 
{ 
	hello::helloworld msg1; 
	msg1.set_id(102); 
	msg1.set_str("hi");

	fstream output("./log", ios::out | ios::trunc | ios::binary);
	if (!msg1.SerializeToOstream(&output)) {
		cout << "Failed to write msg." << endl;
		return -1;
	}
	return 0;
}
