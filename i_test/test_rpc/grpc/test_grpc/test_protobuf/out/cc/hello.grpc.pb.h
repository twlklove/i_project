// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: hello.proto
#ifndef GRPC_hello_2eproto__INCLUDED
#define GRPC_hello_2eproto__INCLUDED

#include "hello.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace hello {

class MyService final {
 public:
  static constexpr char const* service_full_name() {
    return "hello.MyService";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Fool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::hello::MyResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>> AsyncFool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>>(AsyncFoolRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>> PrepareAsyncFool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>>(PrepareAsyncFoolRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      virtual void Fool(::grpc::ClientContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void Fool(::grpc::ClientContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response, ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() { return nullptr; }
    class async_interface* experimental_async() { return async(); }
   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>* AsyncFoolRaw(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::hello::MyResponse>* PrepareAsyncFoolRaw(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status Fool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::hello::MyResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>> AsyncFool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>>(AsyncFoolRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>> PrepareAsyncFool(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>>(PrepareAsyncFoolRaw(context, request, cq));
    }
    class async final :
      public StubInterface::async_interface {
     public:
      void Fool(::grpc::ClientContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response, std::function<void(::grpc::Status)>) override;
      void Fool(::grpc::ClientContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response, ::grpc::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class async* async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>* AsyncFoolRaw(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::hello::MyResponse>* PrepareAsyncFoolRaw(::grpc::ClientContext* context, const ::hello::MyRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Fool_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Fool(::grpc::ServerContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_Fool() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestFool(::grpc::ServerContext* context, ::hello::MyRequest* request, ::grpc::ServerAsyncResponseWriter< ::hello::MyResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Fool<Service > AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_Fool() {
      ::grpc::Service::MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::hello::MyRequest, ::hello::MyResponse>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::hello::MyRequest* request, ::hello::MyResponse* response) { return this->Fool(context, request, response); }));}
    void SetMessageAllocatorFor_Fool(
        ::grpc::MessageAllocator< ::hello::MyRequest, ::hello::MyResponse>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::hello::MyRequest, ::hello::MyResponse>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* Fool(
      ::grpc::CallbackServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/)  { return nullptr; }
  };
  typedef WithCallbackMethod_Fool<Service > CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_Fool() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_Fool() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestFool(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_Fool() {
      ::grpc::Service::MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->Fool(context, request, response); }));
    }
    ~WithRawCallbackMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* Fool(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Fool : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_Fool() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::hello::MyRequest, ::hello::MyResponse>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::hello::MyRequest, ::hello::MyResponse>* streamer) {
                       return this->StreamedFool(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_Fool() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Fool(::grpc::ServerContext* /*context*/, const ::hello::MyRequest* /*request*/, ::hello::MyResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedFool(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::hello::MyRequest,::hello::MyResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Fool<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_Fool<Service > StreamedService;
};

}  // namespace hello


#endif  // GRPC_hello_2eproto__INCLUDED
