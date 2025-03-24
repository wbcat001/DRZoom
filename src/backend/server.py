# pca


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.decomposition import PCA
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
# Generate dummy data (1000 rows, 10 dimensions)
data = np.random.rand(1000, 10)

class PCARequest(BaseModel):
    filter: List[int]

@app.post("/pca/update")
async def pca(request: PCARequest):
    # Filter the data using the provided indices
    filter_array = np.array(request.filter)
    data_filtered = data[filter_array]
    print(request.filter[:10])
    
    # Perform PCA to reduce the data to 2 dimensions
    pca = PCA(n_components=2)
    result = pca.fit_transform(data_filtered).tolist()
    # filterをインデックスとしてresultに追加して zip result, filter
    result = [{"index": i, "data": d} for i, d in zip(request.filter, result)]    
    
    return {"data":result}

class InitRequest(BaseModel):
    options: str
@app.post("/pca/init")
async def pca_init(InitRequest: InitRequest):
    pca = PCA(n_components=2)
    result = pca.fit_transform(data).tolist()
    # index をふる
    result = [{"index": i, "data": d} for i, d in enumerate
    (result)]
    print(result[:10])


    return {"data":result}
## 実行コマンド
# uvicorn server:app --reload
# http://

@app.get("/")
async def root():
    return {"message": "Hello World"}