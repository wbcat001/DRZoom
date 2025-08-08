from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to init layout
@app.post("/init")
async def init_layout():
    # generate random data for demonstration
    # type data: (x, y, label)[]
    data = [
        {
            "x": np.random.rand(),
            "y": np.random.rand(),
            "label": f"Point {i}"
        }
        for i in range(100)
    ]

    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)