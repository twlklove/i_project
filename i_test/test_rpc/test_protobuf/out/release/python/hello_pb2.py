# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hello.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='hello.proto',
  package='hello',
  syntax='proto3',
  serialized_options=b'\200\001\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0bhello.proto\x12\x05hello\"2\n\nhelloworld\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0b\n\x03str\x18\x02 \x01(\t\x12\x0b\n\x03opt\x18\x03 \x01(\x05\"(\n\tMyRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\r\n\x05times\x18\x02 \x01(\x05\"*\n\nMyResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\x08\x32\x38\n\tMyService\x12+\n\x04\x46ool\x12\x10.hello.MyRequest\x1a\x11.hello.MyResponseB\x03\x80\x01\x01\x62\x06proto3'
)




_HELLOWORLD = _descriptor.Descriptor(
  name='helloworld',
  full_name='hello.helloworld',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='hello.helloworld.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='str', full_name='hello.helloworld.str', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='opt', full_name='hello.helloworld.opt', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=72,
)


_MYREQUEST = _descriptor.Descriptor(
  name='MyRequest',
  full_name='hello.MyRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='hello.MyRequest.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='times', full_name='hello.MyRequest.times', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=74,
  serialized_end=114,
)


_MYRESPONSE = _descriptor.Descriptor(
  name='MyResponse',
  full_name='hello.MyResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='hello.MyResponse.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='result', full_name='hello.MyResponse.result', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=158,
)

DESCRIPTOR.message_types_by_name['helloworld'] = _HELLOWORLD
DESCRIPTOR.message_types_by_name['MyRequest'] = _MYREQUEST
DESCRIPTOR.message_types_by_name['MyResponse'] = _MYRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

helloworld = _reflection.GeneratedProtocolMessageType('helloworld', (_message.Message,), {
  'DESCRIPTOR' : _HELLOWORLD,
  '__module__' : 'hello_pb2'
  # @@protoc_insertion_point(class_scope:hello.helloworld)
  })
_sym_db.RegisterMessage(helloworld)

MyRequest = _reflection.GeneratedProtocolMessageType('MyRequest', (_message.Message,), {
  'DESCRIPTOR' : _MYREQUEST,
  '__module__' : 'hello_pb2'
  # @@protoc_insertion_point(class_scope:hello.MyRequest)
  })
_sym_db.RegisterMessage(MyRequest)

MyResponse = _reflection.GeneratedProtocolMessageType('MyResponse', (_message.Message,), {
  'DESCRIPTOR' : _MYRESPONSE,
  '__module__' : 'hello_pb2'
  # @@protoc_insertion_point(class_scope:hello.MyResponse)
  })
_sym_db.RegisterMessage(MyResponse)


DESCRIPTOR._options = None

_MYSERVICE = _descriptor.ServiceDescriptor(
  name='MyService',
  full_name='hello.MyService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=160,
  serialized_end=216,
  methods=[
  _descriptor.MethodDescriptor(
    name='Fool',
    full_name='hello.MyService.Fool',
    index=0,
    containing_service=None,
    input_type=_MYREQUEST,
    output_type=_MYRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MYSERVICE)

DESCRIPTOR.services_by_name['MyService'] = _MYSERVICE

# @@protoc_insertion_point(module_scope)
