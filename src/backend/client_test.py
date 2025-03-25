import requests
# use api for pca
resposne = requests.post("http://localhost:8000/init", json={"options":"pca"})
print(resposne.json())
response = requests.post("http://localhost:8000/update", json={"filter":[1,2,3]})
print(response.json())

# root
response = requests.get("http://localhost:8000/")
print(response.json())