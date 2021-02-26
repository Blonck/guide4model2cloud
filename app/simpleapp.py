import logging
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI

from input import ClientDataSimple


# necessary when started from command line via uvicorn,
# can be removed as soon as code is packed into package
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


app = FastAPI()

encoder_file = 'model/simple_enc.pkl'
model_file = 'model/simple_rf.pkl'

logging.debug(f'Loading model from {model_file}')
model = CustomUnpickler(open(model_file, 'rb')).load()
encoder = CustomUnpickler(open(encoder_file, 'rb')).load()


@app.get('/')
def index():
    return {'message': 'Classifier for telemarketing'}


@app.post('/predict')
def predict(data: ClientDataSimple):
    # transform dict to pandas DataFrame
    data_as_dict = {key: [value] for key, value in data.dict().items()}
    df = pd.DataFrame.from_dict(data_as_dict)
    x = encoder.transform(df)
    preds = model.predict(x)
    preds = pd.Series(preds)

    # transform result back to original presentation
    preds = preds.map({0: 'no', 1: 'yes'})

    return preds.to_json(orient='records')


if __name__ == '__main__':
    logging.info('Starting uvicorn')
    uvicorn.run(app, host='0.0.0.0', port=8000)
