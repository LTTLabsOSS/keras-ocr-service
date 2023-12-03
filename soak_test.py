import requests

# URL of the endpoint
url = "http://localhost:5000/process"

# File path of the picture you want to upload
image_path = "images/1.png"

# Additional text attribute
word_attribute = "nope"

# Prepare the data
files = {'file': open(image_path, 'rb')}
data = {'word': word_attribute}

# Send the POST request
response = requests.post(url, files=files, data=data)

# Check the response
if response.status_code == 200:
    print("POST request successful!")
else:
    print(f"POST request failed with status code {response.status_code}: {response.text}")