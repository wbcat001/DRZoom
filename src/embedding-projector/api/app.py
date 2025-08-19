from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import UploadFile
import uvicorn
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to say hello
@app.get("/hello")
async def hello():
    return {"message": "Hello, World!"}

# Endpoint to init layout
@app.post("/init")
async def init_layout():
    # generate random data for demonstration
    # type data: (x, y, label)[]

    # generate (100, 100)
    sample_data = np.random.rand(100, 100)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(sample_data)
    data = [
        {
            "x": reduced_data[i, 0],
            "y": reduced_data[i, 1],
            "label": f"Point {i}"
        }
        for i in range(100)
    ]

    return data


class DataManager:
    def __init__(self):
        self.metadata = None
        self.data = None
        self.file_names = {
            "data": "data.npy",
            "metadata": "metadata.csv"
        }

    def load_data(self, dir_path: str):
        print(os.path.join(dir_path, self.file_names["data"]))
        self.data = np.load(os.path.join(dir_path, self.file_names["data"]))
        self.metadata = pd.read_csv(os.path.join(dir_path, self.file_names["metadata"]))

    def is_data_loaded(self) -> bool:
        return self.data is not None and self.data.size > 0 and self.metadata is not None


data_manager = DataManager()


@app.post("/init/{file_name}")
async def init_layout_with_file(file_name: str):
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
    file_dir = os.path.join(script_dir, file_name)
    print(f"Loading data from: {file_dir}")

    if not os.path.exists(file_dir):
        return JSONResponse(
            status_code=404,
            content={"message": "File not found."}
        )
    
    data_manager.load_data(file_dir)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_manager.data)

    result = [
        {
            "x": reduced_data[i, 0],
            "y": reduced_data[i, 1],
            "index": i,
            "label": f"Point {i}"
        }
        for i in range(reduced_data.shape[0])
    ]
    return result

@app.post("/files/load")
async def load_data(file_name: str):
    # Load data from a file
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
    file_path = os.path.join(script_dir, file_name)

    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"message": "File not found."}
        )
    
    # read csv 
    df = pd.read_csv(file_path)

@app.get("/files/list")
async def list_files():
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")

    files = os.listdir(script_dir)
    return {"files": files}

@app.post("/files/upload")
async def upload_file(file: UploadFile):
    
    # check file type: csv or other
    file_type = file.content_type
    if file_type != "text/csv":
        print(f"Invalid file type: {file_type}")
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid file type. Please upload a CSV file."}
        )

    df = pd.read_csv(file.file)

    # save file
    file_name = file.filename
    if not file_name:
        return JSONResponse(
            status_code=400,
            content={"message": "File name is required."}
        )

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
    df.to_csv(os.path.join(script_dir, file_name), index=False)

    return JSONResponse(
        status_code=200,
        content={"message": "File uploaded successfully.",
                "file_name": file_name}
    )



# Select Interaction
# # type data: (id, label, distance)[]
@app.post("/select/{index}")
async def select_point(index: int):
    print(f"Selecting point at index: {index}")
    print(f"Data loaded: {data_manager.data}")

    # if not data_manager.is_data_loaded():
    #     return JSONResponse(
    #         status_code=404,
    #         content={"message": "No data loaded."}
    #     )
    
    # if not index < len(data_manager.data):
    #     return JSONResponse(
    #         status_code=404,
    #         content={"message": "Index out of range."}
    #     )
    
    # calc distance 
    top_n = 30
    target_point = data_manager.data[index]
    # distances(cosine distance)
    distances = 1 - np.dot(data_manager.data, target_point) / (np.linalg.norm(data_manager.data, axis=1) * np.linalg.norm(target_point))

    sorted_indices = np.argsort(distances)
    closest_indices = sorted_indices[:top_n]

    closest_points = [{"index": int(i), 
                       "label": int(i), 
                       "distance": float(distances[i])} for i in closest_indices if i != index]

    

    return {"closest_points": closest_points,
            "method": "cosine"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")