import requests

BASE_URL = "http://127.0.0.1:8000"

# Test the root endpoint
def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("Root Response:", response.json())

# Test /config
def test_get_config():
    response = requests.get(f"{BASE_URL}/config")
    print("Config Response:", response.json())

# Test /dimension_reduce/init
def test_dimension_reduce_init():
    data = {"options": "test_option"}
    response = requests.post(f"{BASE_URL}/init", json=data)
    print("Dimension Reduce Init Response:", response)

# Test /dimension_reduce/update
def test_dimension_reduce_update():
    data = {"filter": [0, 1, 2, 3, 4]}
    response = requests.post(f"{BASE_URL}/update", json=data)
    print("Dimension Reduce Update Response:", response)
"""
@app.get("/test")
async def test():
    try:
        main_handler.update([1,2,3])
    except Exception as e:
        print(e)
    return {"message": "test"}
"""
def test_test():
    response = requests.get(f"{BASE_URL}/test")
    print("Test Response:", response.json())

# Run tests
if __name__ == "__main__":
    test_root()
    test_get_config()
    test_dimension_reduce_init()
    test_dimension_reduce_update()
    test_test()
