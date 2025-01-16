from flask import Flask, request, jsonify
from rag_pipeline import rag_chain
import os
import json

# Load configurations from config.json
def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config()

# Configuration
if config is None:
    exit()

# Load RAG Chain
rag = rag_chain()
API_KEY = config.get('api_key', 'YOUR_API_KEY')

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        #Get headers
        headers = request.headers
        api_key = headers.get('X-API-KEY')

        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Unauthorised: Invalid or missing API Key'}), 401

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400

        query = data['query']
        reponse = rag.invoke(query)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if __name__ == '__main__':
        app.run(debug=True, port=5001)
