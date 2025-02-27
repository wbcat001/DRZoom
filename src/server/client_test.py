import requests
# use api for pca
resposne = requests.post("http://localhost:8000/pca/init",json={"options":"test"})
print(resposne.json())
response = requests.post("http://localhost:8000/pca/update", json={"filter":[1,2,3]})
print(response.json())