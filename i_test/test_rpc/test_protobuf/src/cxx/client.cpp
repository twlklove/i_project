#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "hello.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using hello::MyService;
using hello::MyResponse;
using hello::MyRequest;

class Client {
 public:
  Client(std::shared_ptr<Channel> channel) : stub_(MyService::NewStub(channel)) {}

  std::string Fool(const std::string& user) {
    // Data we are sending to the server.
    MyRequest request;
    request.set_text(user);

    MyResponse response;
    ClientContext context;

    Status status = stub_->Fool(&context, request, &response);

    if (status.ok()) {
      return response.text();
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<MyService::Stub> stub_;
};

int main(int argc, char** argv) {
  std::string target_str;
  std::string arg_str("--target");
  if (argc > 1) {
    std::string arg_val = argv[1];
    size_t start_pos = arg_val.find(arg_str);
    if (start_pos != std::string::npos) {
      start_pos += arg_str.size();
      if (arg_val[start_pos] == '=') {
        target_str = arg_val.substr(start_pos + 1);
      } else {
        std::cout << "The only correct argument syntax is --target=" << std::endl;
        return 0;
      }
    } else {
      std::cout << "The only acceptable argument is --target=" << std::endl;
      return 0;
    }
  } else {
    target_str = "localhost:50051";
  }

  Client client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
  std::string user("world");
  std::string response = client.Fool(user);
  std::cout << "Client received: " << response << std::endl;

  return 0;
}
