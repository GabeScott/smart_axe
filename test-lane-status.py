import requests
import json

x = requests.get('https://usaxeclub.com/checkAD.php?loc=1&lane=0')
print(x.text == "0")

