// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: hello.proto

#include "hello.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace hello {
constexpr helloworld::helloworld(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : str_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , id_(0)
  , opt_(0){}
struct helloworldDefaultTypeInternal {
  constexpr helloworldDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~helloworldDefaultTypeInternal() {}
  union {
    helloworld _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT helloworldDefaultTypeInternal _helloworld_default_instance_;
constexpr MyRequest::MyRequest(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : id_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , text_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , times_(0){}
struct MyRequestDefaultTypeInternal {
  constexpr MyRequestDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~MyRequestDefaultTypeInternal() {}
  union {
    MyRequest _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT MyRequestDefaultTypeInternal _MyRequest_default_instance_;
constexpr MyResponse::MyResponse(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : id_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , text_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , result_(false){}
struct MyResponseDefaultTypeInternal {
  constexpr MyResponseDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~MyResponseDefaultTypeInternal() {}
  union {
    MyResponse _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT MyResponseDefaultTypeInternal _MyResponse_default_instance_;
}  // namespace hello
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_hello_2eproto[3];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_hello_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_hello_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_hello_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::hello::helloworld, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::hello::helloworld, id_),
  PROTOBUF_FIELD_OFFSET(::hello::helloworld, str_),
  PROTOBUF_FIELD_OFFSET(::hello::helloworld, opt_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::hello::MyRequest, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::hello::MyRequest, id_),
  PROTOBUF_FIELD_OFFSET(::hello::MyRequest, text_),
  PROTOBUF_FIELD_OFFSET(::hello::MyRequest, times_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::hello::MyResponse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::hello::MyResponse, id_),
  PROTOBUF_FIELD_OFFSET(::hello::MyResponse, text_),
  PROTOBUF_FIELD_OFFSET(::hello::MyResponse, result_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::hello::helloworld)},
  { 9, -1, -1, sizeof(::hello::MyRequest)},
  { 18, -1, -1, sizeof(::hello::MyResponse)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::hello::_helloworld_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::hello::_MyRequest_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::hello::_MyResponse_default_instance_),
};

const char descriptor_table_protodef_hello_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\013hello.proto\022\005hello\"2\n\nhelloworld\022\n\n\002id"
  "\030\001 \001(\005\022\013\n\003str\030\002 \001(\t\022\013\n\003opt\030\003 \001(\005\"4\n\tMyRe"
  "quest\022\n\n\002id\030\001 \001(\t\022\014\n\004text\030\002 \001(\t\022\r\n\005times"
  "\030\003 \001(\005\"6\n\nMyResponse\022\n\n\002id\030\001 \001(\t\022\014\n\004text"
  "\030\002 \001(\t\022\016\n\006result\030\003 \001(\01028\n\tMyService\022+\n\004F"
  "ool\022\020.hello.MyRequest\032\021.hello.MyResponse"
  "B\003\200\001\000b\006proto3"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_hello_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_hello_2eproto = {
  false, false, 253, descriptor_table_protodef_hello_2eproto, "hello.proto", 
  &descriptor_table_hello_2eproto_once, nullptr, 0, 3,
  schemas, file_default_instances, TableStruct_hello_2eproto::offsets,
  file_level_metadata_hello_2eproto, file_level_enum_descriptors_hello_2eproto, file_level_service_descriptors_hello_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_hello_2eproto_getter() {
  return &descriptor_table_hello_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_hello_2eproto(&descriptor_table_hello_2eproto);
namespace hello {

// ===================================================================

class helloworld::_Internal {
 public:
};

helloworld::helloworld(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:hello.helloworld)
}
helloworld::helloworld(const helloworld& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  str_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_str().empty()) {
    str_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_str(), 
      GetArenaForAllocation());
  }
  ::memcpy(&id_, &from.id_,
    static_cast<size_t>(reinterpret_cast<char*>(&opt_) -
    reinterpret_cast<char*>(&id_)) + sizeof(opt_));
  // @@protoc_insertion_point(copy_constructor:hello.helloworld)
}

void helloworld::SharedCtor() {
str_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&id_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&opt_) -
    reinterpret_cast<char*>(&id_)) + sizeof(opt_));
}

helloworld::~helloworld() {
  // @@protoc_insertion_point(destructor:hello.helloworld)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void helloworld::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  str_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void helloworld::ArenaDtor(void* object) {
  helloworld* _this = reinterpret_cast< helloworld* >(object);
  (void)_this;
}
void helloworld::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void helloworld::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void helloworld::Clear() {
// @@protoc_insertion_point(message_clear_start:hello.helloworld)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  str_.ClearToEmpty();
  ::memset(&id_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&opt_) -
      reinterpret_cast<char*>(&id_)) + sizeof(opt_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* helloworld::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // int32 id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string str = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_str();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "hello.helloworld.str"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 opt = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          opt_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* helloworld::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:hello.helloworld)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 id = 1;
  if (this->_internal_id() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_id(), target);
  }

  // string str = 2;
  if (!this->_internal_str().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_str().data(), static_cast<int>(this->_internal_str().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "hello.helloworld.str");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_str(), target);
  }

  // int32 opt = 3;
  if (this->_internal_opt() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_opt(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:hello.helloworld)
  return target;
}

size_t helloworld::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:hello.helloworld)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string str = 2;
  if (!this->_internal_str().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_str());
  }

  // int32 id = 1;
  if (this->_internal_id() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_id());
  }

  // int32 opt = 3;
  if (this->_internal_opt() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_opt());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData helloworld::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    helloworld::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*helloworld::GetClassData() const { return &_class_data_; }

void helloworld::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<helloworld *>(to)->MergeFrom(
      static_cast<const helloworld &>(from));
}


void helloworld::MergeFrom(const helloworld& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:hello.helloworld)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_str().empty()) {
    _internal_set_str(from._internal_str());
  }
  if (from._internal_id() != 0) {
    _internal_set_id(from._internal_id());
  }
  if (from._internal_opt() != 0) {
    _internal_set_opt(from._internal_opt());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void helloworld::CopyFrom(const helloworld& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:hello.helloworld)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool helloworld::IsInitialized() const {
  return true;
}

void helloworld::InternalSwap(helloworld* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &str_, lhs_arena,
      &other->str_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(helloworld, opt_)
      + sizeof(helloworld::opt_)
      - PROTOBUF_FIELD_OFFSET(helloworld, id_)>(
          reinterpret_cast<char*>(&id_),
          reinterpret_cast<char*>(&other->id_));
}

::PROTOBUF_NAMESPACE_ID::Metadata helloworld::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_hello_2eproto_getter, &descriptor_table_hello_2eproto_once,
      file_level_metadata_hello_2eproto[0]);
}

// ===================================================================

class MyRequest::_Internal {
 public:
};

MyRequest::MyRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:hello.MyRequest)
}
MyRequest::MyRequest(const MyRequest& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_id().empty()) {
    id_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_id(), 
      GetArenaForAllocation());
  }
  text_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_text().empty()) {
    text_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_text(), 
      GetArenaForAllocation());
  }
  times_ = from.times_;
  // @@protoc_insertion_point(copy_constructor:hello.MyRequest)
}

void MyRequest::SharedCtor() {
id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
text_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
times_ = 0;
}

MyRequest::~MyRequest() {
  // @@protoc_insertion_point(destructor:hello.MyRequest)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void MyRequest::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  id_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  text_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void MyRequest::ArenaDtor(void* object) {
  MyRequest* _this = reinterpret_cast< MyRequest* >(object);
  (void)_this;
}
void MyRequest::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void MyRequest::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void MyRequest::Clear() {
// @@protoc_insertion_point(message_clear_start:hello.MyRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  id_.ClearToEmpty();
  text_.ClearToEmpty();
  times_ = 0;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* MyRequest::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_id();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "hello.MyRequest.id"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string text = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_text();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "hello.MyRequest.text"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 times = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          times_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* MyRequest::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:hello.MyRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string id = 1;
  if (!this->_internal_id().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_id().data(), static_cast<int>(this->_internal_id().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "hello.MyRequest.id");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_id(), target);
  }

  // string text = 2;
  if (!this->_internal_text().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_text().data(), static_cast<int>(this->_internal_text().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "hello.MyRequest.text");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_text(), target);
  }

  // int32 times = 3;
  if (this->_internal_times() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_times(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:hello.MyRequest)
  return target;
}

size_t MyRequest::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:hello.MyRequest)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string id = 1;
  if (!this->_internal_id().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_id());
  }

  // string text = 2;
  if (!this->_internal_text().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_text());
  }

  // int32 times = 3;
  if (this->_internal_times() != 0) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_times());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData MyRequest::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    MyRequest::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*MyRequest::GetClassData() const { return &_class_data_; }

void MyRequest::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<MyRequest *>(to)->MergeFrom(
      static_cast<const MyRequest &>(from));
}


void MyRequest::MergeFrom(const MyRequest& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:hello.MyRequest)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_id().empty()) {
    _internal_set_id(from._internal_id());
  }
  if (!from._internal_text().empty()) {
    _internal_set_text(from._internal_text());
  }
  if (from._internal_times() != 0) {
    _internal_set_times(from._internal_times());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void MyRequest::CopyFrom(const MyRequest& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:hello.MyRequest)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MyRequest::IsInitialized() const {
  return true;
}

void MyRequest::InternalSwap(MyRequest* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &id_, lhs_arena,
      &other->id_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &text_, lhs_arena,
      &other->text_, rhs_arena
  );
  swap(times_, other->times_);
}

::PROTOBUF_NAMESPACE_ID::Metadata MyRequest::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_hello_2eproto_getter, &descriptor_table_hello_2eproto_once,
      file_level_metadata_hello_2eproto[1]);
}

// ===================================================================

class MyResponse::_Internal {
 public:
};

MyResponse::MyResponse(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:hello.MyResponse)
}
MyResponse::MyResponse(const MyResponse& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_id().empty()) {
    id_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_id(), 
      GetArenaForAllocation());
  }
  text_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_text().empty()) {
    text_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_text(), 
      GetArenaForAllocation());
  }
  result_ = from.result_;
  // @@protoc_insertion_point(copy_constructor:hello.MyResponse)
}

void MyResponse::SharedCtor() {
id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
text_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
result_ = false;
}

MyResponse::~MyResponse() {
  // @@protoc_insertion_point(destructor:hello.MyResponse)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void MyResponse::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  id_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  text_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void MyResponse::ArenaDtor(void* object) {
  MyResponse* _this = reinterpret_cast< MyResponse* >(object);
  (void)_this;
}
void MyResponse::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void MyResponse::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void MyResponse::Clear() {
// @@protoc_insertion_point(message_clear_start:hello.MyResponse)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  id_.ClearToEmpty();
  text_.ClearToEmpty();
  result_ = false;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* MyResponse::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_id();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "hello.MyResponse.id"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string text = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_text();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "hello.MyResponse.text"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // bool result = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          result_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* MyResponse::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:hello.MyResponse)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string id = 1;
  if (!this->_internal_id().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_id().data(), static_cast<int>(this->_internal_id().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "hello.MyResponse.id");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_id(), target);
  }

  // string text = 2;
  if (!this->_internal_text().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_text().data(), static_cast<int>(this->_internal_text().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "hello.MyResponse.text");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_text(), target);
  }

  // bool result = 3;
  if (this->_internal_result() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_result(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:hello.MyResponse)
  return target;
}

size_t MyResponse::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:hello.MyResponse)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string id = 1;
  if (!this->_internal_id().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_id());
  }

  // string text = 2;
  if (!this->_internal_text().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_text());
  }

  // bool result = 3;
  if (this->_internal_result() != 0) {
    total_size += 1 + 1;
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData MyResponse::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    MyResponse::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*MyResponse::GetClassData() const { return &_class_data_; }

void MyResponse::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<MyResponse *>(to)->MergeFrom(
      static_cast<const MyResponse &>(from));
}


void MyResponse::MergeFrom(const MyResponse& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:hello.MyResponse)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_id().empty()) {
    _internal_set_id(from._internal_id());
  }
  if (!from._internal_text().empty()) {
    _internal_set_text(from._internal_text());
  }
  if (from._internal_result() != 0) {
    _internal_set_result(from._internal_result());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void MyResponse::CopyFrom(const MyResponse& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:hello.MyResponse)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MyResponse::IsInitialized() const {
  return true;
}

void MyResponse::InternalSwap(MyResponse* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &id_, lhs_arena,
      &other->id_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &text_, lhs_arena,
      &other->text_, rhs_arena
  );
  swap(result_, other->result_);
}

::PROTOBUF_NAMESPACE_ID::Metadata MyResponse::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_hello_2eproto_getter, &descriptor_table_hello_2eproto_once,
      file_level_metadata_hello_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace hello
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::hello::helloworld* Arena::CreateMaybeMessage< ::hello::helloworld >(Arena* arena) {
  return Arena::CreateMessageInternal< ::hello::helloworld >(arena);
}
template<> PROTOBUF_NOINLINE ::hello::MyRequest* Arena::CreateMaybeMessage< ::hello::MyRequest >(Arena* arena) {
  return Arena::CreateMessageInternal< ::hello::MyRequest >(arena);
}
template<> PROTOBUF_NOINLINE ::hello::MyResponse* Arena::CreateMaybeMessage< ::hello::MyResponse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::hello::MyResponse >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
