import os
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

# Define the upload folder
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

# Load the trained machine learning model
model_filename = "modelo.pkl"
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions using the loaded model
def make_predictions(data):
    # Replace this with the actual prediction logic for your model
    # For example, if you're using scikit-learn, you can use model.predict(data)
    predictions = model.predict(data)
    return predictions

@app.post("/predict/")
async def predict(data: list):
    try:
        # Ensure the input data is in the expected format (e.g., a list)
        # You may need to preprocess the input data based on your model's requirements
        predictions = make_predictions(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)