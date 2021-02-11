from socketIO_client_nexus import SocketIO
HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)


while True:
    x = input("Enter x: ")
    y = input("Enter y: ")
    data = {'lane':0, 'x':x, 'y':y, 'width':2, 'height':50}
    HIT_SOCKET.emit("test hit", data)