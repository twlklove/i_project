CC = g++
CFLAGS = -std=c++11
release_dir = release

all: init Server.o Client.o
	$(CC) $(CFLAGS) ServerMain.cpp Server.o -o $(release_dir)/chatroom_server
	$(CC) $(CFLAGS) ClientMain.cpp Client.o -o $(release_dir)/chatroom_client

init:
	$(shell mkdir $(release_dir) >/dev/null 2>&1)

Server.o: Server.cpp Server.h Common.h
	$(CC) $(CFLAGS) -c Server.cpp

Client.o: Client.cpp Client.h Common.h
	$(CC) $(CFLAGS) -c Client.cpp

clean:
	rm -rf *.o *~ $(release_dir)
