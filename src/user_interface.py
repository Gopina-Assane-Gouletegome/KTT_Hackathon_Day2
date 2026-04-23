import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("model.onnx")

classes = [
    "healthy",
    "maize_rust",
    "maize_blight",
    "cassava_mosaic",
    "bean_spot"
]

def predict(image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    probs = np.array(outputs[0])[0]
    idx = np.argmax(probs)

    return {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "top3": sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )[:3]
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="🌱 Crop Disease Classifier",
    description="Upload a leaf image to detect crop disease"
)

demo.launch(True)