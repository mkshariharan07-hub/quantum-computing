import requests

url = "http://127.0.0.1:5000/predict"

files = {
    "image": open("D:\Downloads\images.jpg", "rb")   # put any test image here
}

response = requests.post(url, files=files)

print(response.json())