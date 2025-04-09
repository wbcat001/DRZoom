from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from services.main_manager import MainManager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

main_manager = MainManager()


# 型定義

class ZoomRequest(BaseModel):
    filter: List[int]

class InitRequest(BaseModel):
    options: str

@app.post("/init")
async def init(init_request: InitRequest):
    try:
        global main_manager
        position_data = main_manager.init_layout()
        data = [{"index": i, "data": d} for i, d in enumerate(position_data.tolist())]
        return {"data": data}
    except Exception as e:
        print(e)
        # internal_server_error
        return {"error": "Internal Sserver Error", "status_code": 500}
    

@app.post("/zoom")
async def zoom(request: ZoomRequest):
    try:
        print(request.filter)
        global main_manager
        position_data = main_manager.update_layout(request.filter)
        data = [{"index": i, "data": d} for i, d in enumerate(position_data.tolist())]
        print(f"length: {len(data)}")
        return {"data": data}
    except Exception as e:
        print(e)
        # internal_server_error
        return {"error": "Internal Sserver Error", "status_code": 500}

@app.post("/update_config")
async def update_config(request: ZoomRequest):
    pass

@app.get("/test")
async def test():
    return {"message": "Hello World."}

"""
uvicorn main:app --reload
"""