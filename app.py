# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
from g4f.client import Client
import requests
import os

app = Flask(__name__)

# Configure CORS to allow specific origins
CORS(app, resources={r"/*": {"origins": "*"}})
client = Client()

# Route for processing image from URL
@app.route('/analyze_url', methods=['POST'])
def analyze_image_url():
    try:
        data = request.get_json()
        image_url = data.get('url')
        prompt = data.get('prompt', 'What are on this image?')
        
        if not image_url:
            return jsonify({'error': 'No URL provided'}), 400
            
        remote_image = requests.get(image_url, stream=True).content
        
        response = client.chat.completions.create(
            model=g4f.models.blackboxai_pro,
            messages=[{"role": "user", "content": prompt}],
            image=remote_image
        )
        
        result = response.choices[0].message.content
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for processing local image upload
@app.route('/analyze_upload', methods=['POST'])
def analyze_image_upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        image_data = image_file.read()
        prompt = request.form.get('prompt', 'What are on this image?')
        
        response = client.chat.completions.create(
            model=g4f.models.blackboxai_pro,
            messages=[{"role": "user", "content": prompt}],
            image=image_data
        )
        
        result = response.choices[0].message.content
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment variable PORT if available (for Railway), default to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=port, debug=True)
