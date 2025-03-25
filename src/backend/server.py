


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.decomposition import PCA
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from handler.main_handler import MainHandler


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

main_handler = MainHandler()
print(main_handler.get_initial_data()[:10])
print(main_handler.get_config())
print(main_handler.update([1,2,3])[:10])

class DimmensionReduceRequest(BaseModel):
    filter: List[int]

class InitRequest(BaseModel):
    options: str

@app.post("/init")
async def dimension_reduce_init(init_request: InitRequest):
    try:
        global main_handler
        position_data = main_handler.get_initial_data()
        print(position_data[:10])

        data = [{"index": i, "data": d} for i, d in enumerate(position_data)]
        print(data[:10])
    except Exception as e:
        print(e)
        return {"data": []}
    return {"data": data}

@app.post("/update")
async def dimension_reduce_update(request: DimmensionReduceRequest):
    try:
        global main_handler
        position_data = main_handler.update(request.filter)
        print(request.filter[:10])
        result = [{"index": i, "data": d} for i, d in zip(request.filter, position_data[0])]
    except Exception as e:
        print(e)
        return {"data": []}
    return {"data": result}

@app.get("/config")
async def get_config():
    return main_handler.get_config()

@app.get("/")
async def root():
    return {"message": "Hello World. this is DRZoom API"}

@app.get("/test")
async def test_endpoint():

    return {"message": "test"}