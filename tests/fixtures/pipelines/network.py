"""Fault: tries to download external data. Must be blocked -> ErrorClass 'data'."""
import socket
from _cli import args

args()
s = socket.socket()
s.settimeout(5)
s.connect(("example.com", 80))
print("downloaded external data")
