import numpy as np
import os
import pandas as pd
import pandera as pa
from pandera import Column
from pandera.typing import DataFrame
from pandera.typing import Series

# メタデータのスキーマ定義（panderaを使用）
class MetadataSchema(pa.DataFrameModel):
    Index: Series[int] = pa.Field(ge=1)
    Chapter: Series[int] = pa.Field(ge=1)
    Content: Series[str] = pa.Field()
@pa.check_types
def validate_metadata(metadata: DataFrame) -> DataFrame:
    return MetadataSchema.validate(metadata)
    

df = pd.DataFrame({
    "Index": [1],
    "Chapter": [1],
    "Content": ["test"],
})

validated = MetadataSchema.validate(df)
print(validated)
print(validate_metadata(df))


