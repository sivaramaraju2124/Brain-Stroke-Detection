from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import requests
import json
import base64

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "phase2_kesava.h5")

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# MEMORY OPTIMIZATION FOR RENDER (512MB RAM)
# ==========================================
# Instead of creating the grad_model on every image upload (which crashes the server), 
# we create it ONCE globally when the app starts.
last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        last_conv_layer = layer.name
        break

if last_conv_layer:
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
else:
    grad_model = None

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

# Grad-CAM Implementation
def generate_gradcam(img_path, grad_model):
    if not grad_model:
        return None
    try:
        # Preprocess image
        img = preprocess_image(img_path)
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Load original image
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Save Grad-CAM image
        gradcam_filename = "gradcam_" + os.path.basename(img_path)
        gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        
        # Clear keras session to free up memory immediately after processing
        tf.keras.backend.clear_session()
        
        return gradcam_path
        
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

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
            pred = model.predict(img, verbose=0)[0][0]

            print("RAW MODEL OUTPUT:", pred)

            # class 0 = hemorrhage, class 1 = normal
            if pred < 0.5:
                prediction = "Hemorrhagic Stroke Detected"
                confidence = (1 - pred) * 100
            else:
                prediction = "Normal CT Scan"
                confidence = pred * 100

            img_path = save_path
            
            # Generate Grad-CAM
            gradcam_path = generate_gradcam(save_path, grad_model)

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