automic_file=test_automic.cpp
mutex_file=test_mutex.cpp 
con_variable_file=test_con_variable.cpp  
 
run_automic=run_automic 
run_mutex=run_mutex
run_con_variable=run_con_variable
run_xx=$(run_automic) $(run_mutex) $(run_con_variable)

CXXFLAGS=-g -O2 -std=c++11
LDFLAGS=-lpthread

all:clean run

run:$(run_automic) $(run_mutex) $(run_con_variable)

$(run_automic):$(automic_file)
	g++ $< $(CXXFLAGS) $(LDFLAGS) -o $@

$(run_mutex):$(mutex_file)
	g++ $< $(CXXFLAGS) $(LDFLAGS) -o $@

$(run_con_variable):$(con_variable_file)
	g++ $< $(CXXFLAGS) $(LDFLAGS) -o $@

.PHONY:clean
clean:
	rm -rf $(run_xx) *~
