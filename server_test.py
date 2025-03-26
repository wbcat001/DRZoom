# make fastapi server test

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import List
import numpy as np



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "hello, world!!!!!"}

@app.get("/number")
def read_number():
    return {"number": np.random.rand()}

@app.get("/data")
def read_data():
    data = np.random.rand(1000, 10)
    return {"data": data.tolist()}