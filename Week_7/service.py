import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

from pydantic import BaseModel



model_ref = bentoml.sklearn.get("mlzoomcamp_homework:latest")

model_ref_runner = model_ref.to_runner()

svc = bentoml.Service("service", runners=[model_ref_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(UserProfile):
    prediction = model_ref_runner.predict.run(UserProfile)
    print("prediction = ", prediction)
    return({"prediction" : prediction})
