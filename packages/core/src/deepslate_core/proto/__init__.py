# Ensure google.protobuf.struct_pb2 is loaded before our proto
from google.protobuf import struct_pb2 as _struct_pb2  # noqa: F401

from .realtime_pb2 import *  # noqa: F403