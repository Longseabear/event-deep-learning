import socket
import threading
from struct import pack

host = '127.0.0.1'
port = 1568
print('afsdaf dafasfdadsf dsafsadfsd asdsad'.split(' ', maxsplit=1))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

try:
    while True:
        data = input()
        length = pack('>Q', len(data))
        client_socket.sendall(length)
        client_socket.sendall(data.encode())
        ack = client_socket.recv(1)
        print('send ack ok')
except Exception as e:
    print(e)
    client_socket.close()
