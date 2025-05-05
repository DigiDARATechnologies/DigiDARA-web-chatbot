import os
from flask import Flask, request, render_template, jsonify
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging
import nltk
from dotenv import load_dotenv

# Download NLTK data
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
    raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
COLLECTION_NAME = "digidara_website_info"

# Initialize Qdrant Client
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info("Qdrant client initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize Qdrant client: %s", e)
    raise

# Initialize Sentence Transformer for embeddings (lazy-loaded)
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        logger.info("Loading SentenceTransformer model...")
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully.")
    return embedder

# Ensure collection exists
def init_collection():
    try:
        embedder = get_embedder()
        vector_size = embedder.get_sentence_embedding_dimension()
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME} with vector size {vector_size}")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
        raise

# Process and upload text file
def process_and_upload(file):
    try:
        # Read file content
        content = file.read().decode('utf-8')
        # Split into sentences
        sentences = nltk.sent_tokenize(content)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 500:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        points = []
        idx = qdrant_client.count(collection_name=COLLECTION_NAME).count
        
        embedder = get_embedder()
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue  # Skip empty chunks
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
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                logger.info(f"Uploaded batch {i // batch_size + 1} of {file.filename} with {len(batch)} chunks to Qdrant.")
            return {"message": f"Successfully uploaded {file.filename} with {len(chunks)} chunks"}
        return {"error": "No valid content to upload"}
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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

if __name__ == '__main__':
    logger.warning("Running in development mode. Use Gunicorn for production.")
    init_collection()
    app.run(host='0.0.0.0', port=5001, debug=True)