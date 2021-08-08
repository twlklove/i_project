// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: first.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_first_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_first_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3008000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3008000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_first_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_first_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_first_2eproto;
namespace lm {
class helloworld;
class helloworldDefaultTypeInternal;
extern helloworldDefaultTypeInternal _helloworld_default_instance_;
}  // namespace lm
PROTOBUF_NAMESPACE_OPEN
template<> ::lm::helloworld* Arena::CreateMaybeMessage<::lm::helloworld>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace lm {

// ===================================================================

class helloworld :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:lm.helloworld) */ {
 public:
  helloworld();
  virtual ~helloworld();

  helloworld(const helloworld& from);
  helloworld(helloworld&& from) noexcept
    : helloworld() {
    *this = ::std::move(from);
  }

  inline helloworld& operator=(const helloworld& from) {
    CopyFrom(from);
    return *this;
  }
  inline helloworld& operator=(helloworld&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const helloworld& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const helloworld* internal_default_instance() {
    return reinterpret_cast<const helloworld*>(
               &_helloworld_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(helloworld* other);
  friend void swap(helloworld& a, helloworld& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline helloworld* New() const final {
    return CreateMaybeMessage<helloworld>(nullptr);
  }

  helloworld* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<helloworld>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const helloworld& from);
  void MergeFrom(const helloworld& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(helloworld* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "lm.helloworld";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_first_2eproto);
    return ::descriptor_table_first_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string str = 2;
  void clear_str();
  static const int kStrFieldNumber = 2;
  const std::string& str() const;
  void set_str(const std::string& value);
  void set_str(std::string&& value);
  void set_str(const char* value);
  void set_str(const char* value, size_t size);
  std::string* mutable_str();
  std::string* release_str();
  void set_allocated_str(std::string* str);

  // int32 id = 1;
  void clear_id();
  static const int kIdFieldNumber = 1;
  ::PROTOBUF_NAMESPACE_ID::int32 id() const;
  void set_id(::PROTOBUF_NAMESPACE_ID::int32 value);

  // int32 opt = 3;
  void clear_opt();
  static const int kOptFieldNumber = 3;
  ::PROTOBUF_NAMESPACE_ID::int32 opt() const;
  void set_opt(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:lm.helloworld)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr str_;
  ::PROTOBUF_NAMESPACE_ID::int32 id_;
  ::PROTOBUF_NAMESPACE_ID::int32 opt_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_first_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// helloworld

// int32 id = 1;
inline void helloworld::clear_id() {
  id_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 helloworld::id() const {
  // @@protoc_insertion_point(field_get:lm.helloworld.id)
  return id_;
}
inline void helloworld::set_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  id_ = value;
  // @@protoc_insertion_point(field_set:lm.helloworld.id)
}

// string str = 2;
inline void helloworld::clear_str() {
  str_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& helloworld::str() const {
  // @@protoc_insertion_point(field_get:lm.helloworld.str)
  return str_.GetNoArena();
}
inline void helloworld::set_str(const std::string& value) {
  
  str_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:lm.helloworld.str)
}
inline void helloworld::set_str(std::string&& value) {
  
  str_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:lm.helloworld.str)
}
inline void helloworld::set_str(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  str_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:lm.helloworld.str)
}
inline void helloworld::set_str(const char* value, size_t size) {
  
  str_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:lm.helloworld.str)
}
inline std::string* helloworld::mutable_str() {
  
  // @@protoc_insertion_point(field_mutable:lm.helloworld.str)
  return str_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* helloworld::release_str() {
  // @@protoc_insertion_point(field_release:lm.helloworld.str)
  
  return str_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void helloworld::set_allocated_str(std::string* str) {
  if (str != nullptr) {
    
  } else {
    
  }
  str_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), str);
  // @@protoc_insertion_point(field_set_allocated:lm.helloworld.str)
}

// int32 opt = 3;
inline void helloworld::clear_opt() {
  opt_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 helloworld::opt() const {
  // @@protoc_insertion_point(field_get:lm.helloworld.opt)
  return opt_;
}
inline void helloworld::set_opt(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  opt_ = value;
  // @@protoc_insertion_point(field_set:lm.helloworld.opt)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace lm

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_first_2eproto
