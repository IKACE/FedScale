# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: job_api.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\rjob_api.proto\x12\x08\x66\x65\x64scale";\n\x0eServerResponse\x12\r\n\x05\x65vent\x18\x01 \x01(\t\x12\x0c\n\x04meta\x18\x02 \x01(\x0c\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c"P\n\x0fRegisterRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x13\n\x0b\x65xecutor_id\x18\x02 \x01(\t\x12\x15\n\rexecutor_info\x18\x03 \x01(\x0c"5\n\x0bPingRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x13\n\x0b\x65xecutor_id\x18\x02 \x01(\t"\x8f\x01\n\x0f\x43ompleteRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x13\n\x0b\x65xecutor_id\x18\x02 \x01(\t\x12\r\n\x05\x65vent\x18\x03 \x01(\t\x12\x0e\n\x06status\x18\x04 \x01(\x08\x12\x0b\n\x03msg\x18\x05 \x01(\t\x12\x13\n\x0bmeta_result\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x61ta_result\x18\x07 \x01(\x0c\x32\xec\x01\n\nJobService\x12H\n\x0f\x43LIENT_REGISTER\x12\x19.fedscale.RegisterRequest\x1a\x18.fedscale.ServerResponse"\x00\x12@\n\x0b\x43LIENT_PING\x12\x15.fedscale.PingRequest\x1a\x18.fedscale.ServerResponse"\x00\x12R\n\x19\x43LIENT_EXECUTE_COMPLETION\x12\x19.fedscale.CompleteRequest\x1a\x18.fedscale.ServerResponse"\x00\x62\x06proto3'
)


_SERVERRESPONSE = DESCRIPTOR.message_types_by_name["ServerResponse"]
_REGISTERREQUEST = DESCRIPTOR.message_types_by_name["RegisterRequest"]
_PINGREQUEST = DESCRIPTOR.message_types_by_name["PingRequest"]
_COMPLETEREQUEST = DESCRIPTOR.message_types_by_name["CompleteRequest"]
ServerResponse = _reflection.GeneratedProtocolMessageType(
    "ServerResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _SERVERRESPONSE,
        "__module__": "job_api_pb2"
        # @@protoc_insertion_point(class_scope:fedscale.ServerResponse)
    },
)
_sym_db.RegisterMessage(ServerResponse)

RegisterRequest = _reflection.GeneratedProtocolMessageType(
    "RegisterRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _REGISTERREQUEST,
        "__module__": "job_api_pb2"
        # @@protoc_insertion_point(class_scope:fedscale.RegisterRequest)
    },
)
_sym_db.RegisterMessage(RegisterRequest)

PingRequest = _reflection.GeneratedProtocolMessageType(
    "PingRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _PINGREQUEST,
        "__module__": "job_api_pb2"
        # @@protoc_insertion_point(class_scope:fedscale.PingRequest)
    },
)
_sym_db.RegisterMessage(PingRequest)

CompleteRequest = _reflection.GeneratedProtocolMessageType(
    "CompleteRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _COMPLETEREQUEST,
        "__module__": "job_api_pb2"
        # @@protoc_insertion_point(class_scope:fedscale.CompleteRequest)
    },
)
_sym_db.RegisterMessage(CompleteRequest)

_JOBSERVICE = DESCRIPTOR.services_by_name["JobService"]
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SERVERRESPONSE._serialized_start = 27
    _SERVERRESPONSE._serialized_end = 86
    _REGISTERREQUEST._serialized_start = 88
    _REGISTERREQUEST._serialized_end = 168
    _PINGREQUEST._serialized_start = 170
    _PINGREQUEST._serialized_end = 223
    _COMPLETEREQUEST._serialized_start = 226
    _COMPLETEREQUEST._serialized_end = 369
    _JOBSERVICE._serialized_start = 372
    _JOBSERVICE._serialized_end = 608
# @@protoc_insertion_point(module_scope)
