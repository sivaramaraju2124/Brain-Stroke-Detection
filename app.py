from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import requests
import json
import base64

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained TFLite model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "phase2_kesava.tflite")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Preprocessing (MATCHES TRAINING)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    gradcam_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            img = preprocess_image(save_path)
            
            # TFLite Inference (Dual Output)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            
            out0 = interpreter.get_tensor(output_details[0]['index'])
            out1 = interpreter.get_tensor(output_details[1]['index'])
            
            # Identify prediction and conv outputs by shape
            if len(out0.shape) == 2:
                pred = out0[0][0]
                conv_outputs = out1
            else:
                pred = out1[0][0]
                conv_outputs = out0

            print("RAW MODEL OUTPUT:", pred)

            # class 0 = hemorrhage, class 1 = normal
            if pred < 0.5:
                prediction = "Hemorrhagic Stroke Detected"
                confidence = (1 - pred) * 100
            else:
                prediction = "Normal CT Scan"
                confidence = pred * 100

            img_path = save_path
            
            # Generate Saliency Map (Activation-only CAM)
            heatmap = np.mean(conv_outputs[0], axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            
            # Load original image
            original_img = cv2.imread(save_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superimpose
            superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
            
            # Save Heatmap image
            gradcam_filename = "gradcam_" + os.path.basename(save_path)
            gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
            cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        gradcam_path=gradcam_path
    )

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # System prompt to restrict chatbot scope
        system_prompt = """You are a medical AI assistant specialized in brain strokes and CT scan interpretation. 
        
Your scope is LIMITED to:
- Explaining hemorrhagic strokes
- Discussing CT scan features (non-diagnostic)
- Clarifying prediction results from AI models
- General educational information about strokes
- Next steps for seeking medical care (NOT treatment advice)

You MUST NOT:
- Provide medical diagnosis or treatment
- Answer questions unrelated to brain strokes
- Give specific medication recommendations
- Replace professional medical consultation

If a question is outside your scope, politely respond: "I can assist only with brain stroke-related questions. For other medical concerns, please consult a healthcare professional."

Always include a disclaimer that this is for educational purposes only."""

        # Call Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            bot_message = result["choices"][0]["message"]["content"]
            return jsonify({"response": bot_message})
        else:
            return jsonify({"error": "Failed to get response from chatbot"}), 500
            
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)