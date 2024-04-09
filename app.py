import fastapi
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
from tensorflow.keras.layers import TextVectorization


loaded_model = load_model('my_complete_model.h5')
# Load the configuration
with open('vectorizer_config.json', 'r') as f:
    config = json.load(f)

# Recreate the TextVectorization layer from the config
text_vectorizer = TextVectorization.from_config(config) 

labels= {0: 'Arrival',
 1: 'Departure',
 2: 'Empty Container Released',
 3: 'Empty Return',
 4: 'Gate In',
 5: 'Gate Out',
 6: 'In-transit',
 7: 'Inbound Terminal',
 8: 'Loaded on Vessel',
 9: 'Off Rail',
 10: 'On Rail',
 11: 'Outbound Terminal',
 12: 'Port In',
 13: 'Port Out',
 14: 'Unloaded on Vessel'}


web=FastAPI()

class modelitem(BaseModel):
    externalstatus: str

@web.post('/')
async def scoring_endpoint(item:modelitem):
    item=item.dict()
    externalstatus=item['externalstatus']
    vectorizedtext=text_vectorizer(externalstatus)
    pos=loaded_model.predict(vectorizedtext)
    pred=np.argmax(pos,axis=1)
    answer=labels[pred]

    return {'internalstatus':answer}