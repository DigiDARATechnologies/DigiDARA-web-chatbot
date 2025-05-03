import os
from flask import Flask, request, render_template, jsonify
from accelerate.utils.memory import clear_device_cache
from timm.data import ImageNetInfo
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging

app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "digidara_website_info"

# Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Sentence Transformer for embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Ensure collection exists
def init_collection():
    try:
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
        raise

# Process and upload text file
def process_and_upload(file):
    try:
        # Read file content
        content = file.read().decode('utf-8')
        # Chunk content into 500-character pieces
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        
        points = []
        idx = qdrant_client.count(collection_name=COLLECTION_NAME).count  # Start ID after existing points
        
        for chunk_idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            points.append(
                models.PointStruct(
                    id=idx + chunk_idx,
                    vector=embedding,
                    payload={
                        "title": f"DigiDara Website Info - {file.filename} Part {chunk_idx + 1}",
                        "content": chunk
                    }
                )
            )
        
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            logger.info(f"Uploaded {file.filename} with {len(chunks)} chunks to Qdrant.")
            return {"message": f"Successfully uploaded {file.filename}"}
        return {"error": "No content to upload"}
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        return {"error": f"Failed to process {file.filename}: {str(e)}"}

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    files = request.files.getlist('file')  # Handle multiple files
    results = []

    for file in files:
        if file.filename == '':
            results.append({"error": "No file selected"})
            continue
        if file and file.filename.endswith('.txt'):
            result = process_and_upload(file)
            results.append(result)
        else:
            results.append({"error": "Only .txt files are supported"})

    return jsonify(results)

if __name__ == '__main__':
    init_collection()
    app.run(host='0.0.0.0', port=5001, debug=True)