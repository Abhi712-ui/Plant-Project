from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from app.inference import load_model, load_class_names, predict

app = FastAPI(title="Plant Disease Classifier")

class_names = load_class_names()
model = load_model(len(class_names))

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "num_classes": len(class_names)
    }
    
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    contents = await file.read()
    
    try:
        image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")
    
    predictions = predict(model, class_names, image)
    return {"predictions": predictions}