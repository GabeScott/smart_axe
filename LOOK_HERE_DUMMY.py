from socketIO_client_nexus import SocketIO
HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)


while True:
    new_coords = input("Enter x y or QUIT to exit")
    if new_coords == 'QUIT':
        keep_trying = False
    else:
        x = new_coords.split(" ")[0]
        y = new_coords.split(" ")[1]


        data = {'lane':0,
                'x':x,
                'y':y,
                'width':2,
                'height':50}
        HIT_SOCKET.emit('test hit', data)