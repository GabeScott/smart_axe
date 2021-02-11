HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)


while True:
	x = input("Enter x: ")
	y = input("Enter y: ")
	data = {'lane':0,
    	    'x':x,
        	'y':y,
        	'width':10,
        	'height':10}
    HIT_SOCKET.emit("test hit", data)