from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
from g4f.client import Client
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
client = Client()

# Route for processing image from URL
@app.route('/analyze_url', methods=['POST'])
def analyze_image_url():
    try:
        data = request.get_json()
        image_url = data.get('url')
        prompt = data.get('prompt', 'What are on this image?')  # Default prompt if none provided
        
        if not image_url:
            return jsonify({'error': 'No URL provided'}), 400
            
        # Get image from URL
        remote_image = requests.get(image_url, stream=True).content
        
        # Analyze with g4f using custom prompt
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
        prompt = request.form.get('prompt', 'What are on this image?')  # Default prompt if none provided
        
        # Analyze with g4f using custom prompt
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
    app.run(debug=True, port=5000)
