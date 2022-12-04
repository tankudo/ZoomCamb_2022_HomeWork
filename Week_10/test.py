import requests
import time


url = "http://localhost:9696/predict"

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
# response = requests.post(url, json=client).json()
while True:
    time.sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)

print(response)