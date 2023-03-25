from typing import Annotated
from main import model_predict
from fastapi import FastAPI, File, UploadFile
import shutil
import random
import string
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def generate_random_string(length):
#     letters = string.ascii_letters
#     result_str = ''.join(random.choice(letters) for i in range(length))
#     return result_str

@app.post("/uploadfile/")
async def create_upload_image(file: UploadFile = File(...)):
    try:
        os.mkdir("uploaded_images")
    except:
        pass
    file_location = f"uploaded_images/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        output = model_predict(file_location)
    except:
        return {"Error": "No such Category Exists. I am still learning!"}
    os.remove(file_location)
    return {"Prediction": output}