# server.py
from socket import *
from fib import fib


def fib_server(address):
    print("Fib server started")
    sock = socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    sock.bind(address)
    sock.listen(5)
    print("Listening")
    while True:
        client, addr = sock.accept()
        print("Connection", addr)
        fib_handler(client)


def fib_handler(client):
    print("Client: ", client)
    print("Waiting for client input")
    client.send("Enter integer: ".encode('ascii'))
    counter = 0
    while True:
        req = client.recv(100)
        counter += 1
        if not req:
            break
        n = int(req)
        print("n: {}".format(n))
        result = fib(n)
        print("result: {}".format(result))
        resp = str(result).encode('ascii') + b'\n'
        client.send(resp)
        client.send("Counter: {}".format(counter).encode('ascii') + b'\n')
        client.send("Enter integer: ".encode('ascii'))
    print("Closed")


fib_server(('', 25000))
