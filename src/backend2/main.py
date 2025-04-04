from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 型定義

class ZoomRequest(BaseModel):
    filter: List[int]

class InitRequest(BaseModel):
    options: str

@app.post("/init")
async def init(init_request: InitRequest):
    pass

@app.post("/zoom")
async def zoom(request: ZoomRequest):
    pass

@app.post("/update_config")
async def update_config(request: ZoomRequest):
    pass

@app.get("/test")
async def test():
    return {"message": "Hello World."}