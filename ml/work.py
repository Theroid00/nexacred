import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
import sys
import time

# --- 1. CONFIGURATION ---
# Update your MongoDB URI with actual credentials when ready for production
MONGO_URI = "mongodb+srv://hetshah05:Hetshahmit05@nexacred.9ndp6ei.mongodb.net/?retryWrites=true&w=majority&appName=nexacred"
DB_NAME = "financial_advice_db"
COLLECTION_NAME = "rbi_guidelines"
PDF_PATH = "MDP2PB9A1F7F3BDAC463EAF1EEE48A43F2F6C.PDF"  # Updated to use the actual PDF file
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # A great starting model

# --- 2. INITIALIZE MODELS AND CLIENTS ---
# Initialize the embedding model
print("Loading sentence-transformer model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize MongoDB client
print("Connecting to MongoDB...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 second timeout
    # Check if connection is successful
    client.server_info()  # Will raise exception if connection fails
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    print("Continuing with extraction, but data won't be stored in MongoDB")
    client = None

# --- 3. TEXT EXTRACTION AND CHUNKING ---
def extract_and_chunk_text(pdf_path):
    """Extracts text from a PDF and chunks it into paragraphs."""
    print(f"Processing PDF: {pdf_path}")
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return []
            
        doc = fitz.open(pdf_path)
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Splitting text into paragraphs. A more advanced splitter could be used here.
            # PyMuPDF's page.get_text() is dynamically available but may not be recognized by linters
            blocks = page.get_text("blocks")  # type: ignore
            for block in blocks:
                # block[4] is the text content
                chunk_text = block[4].strip()
                if len(chunk_text) > 50: # Filter out very short, irrelevant text blocks
                    chunks.append({
                        "text_chunk": chunk_text.replace("\n", " "),
                        "source_document": os.path.basename(pdf_path),
                        "page_number": page_num + 1
                    })
        return chunks
    except fitz.FileDataError:
        print(f"Error: The file at {pdf_path} is not a valid PDF file")
        return []
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

# --- 4. EMBEDDING AND STORAGE ---
def embed_and_store(chunks):
    """Generates embeddings and stores the data in MongoDB Atlas."""
    if not chunks:
        print("No chunks to process.")
        return
        
    print(f"Found {len(chunks)} chunks to process.")
    
    try:
        # Get the text from all chunks for batch processing
        texts_to_embed = [chunk["text_chunk"] for chunk in chunks]
        
        print("Generating embeddings... (This may take a while for large documents)")
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
        
        # Add the embedding to each chunk document
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
            
        # Only attempt to store in MongoDB if client connection was successful
        if client is not None:
            try:
                print("Inserting documents into MongoDB...")
                collection.insert_many(chunks)
                print(f"Successfully inserted {len(chunks)} documents into the '{COLLECTION_NAME}' collection.")
            except Exception as e:
                print(f"Error inserting documents into MongoDB: {e}")
                print("Data was processed but not stored. Consider saving to a file.")
        else:
            print("MongoDB connection not available. Data was processed but not stored.")
            # Optionally save to a file
            # import json
            # with open('processed_data.json', 'w') as f:
            #     json.dump(chunks, f)
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        print("Make sure the sentence-transformers package is properly installed.")

# --- 5. RUN THE PIPELINE ---
if __name__ == "__main__":
    try:
        print(f"Starting process for PDF: {PDF_PATH}")
        
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF file not found at {PDF_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            print("Available files in current directory:")
            for file in os.listdir():
                print(f" - {file}")
            exit(1)
            
        text_chunks = extract_and_chunk_text(PDF_PATH)
        if text_chunks:
            embed_and_store(text_chunks)
        else:
            print("No text chunks were extracted from the PDF.")
            
        print("Process completed.")
        
        print("\nIMPORTANT: After running this, if using MongoDB Atlas,")
        print("go to your MongoDB Atlas UI, find your collection,")
        print("and create a Vector Search Index on the \"embedding\" field.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
# Initialize MongoDB client
print("Connecting to MongoDB...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 second timeout
    # Check if connection is successful
    client.server_info()  # Will raise exception if connection fails
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB")

    # ✅ Debugging line: list databases to confirm connection
    print("Databases in your cluster:", client.list_database_names())

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    print("Continuing with extraction, but data won't be stored in MongoDB")
    client = None
