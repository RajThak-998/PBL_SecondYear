from flask import Flask, request, jsonify
import PyPDF2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# API Keys and Configuration
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY', 'your-default-api-key-here')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model and tokenizer
try:
    model_name = "google/flan-t5-base"  # Using base model for lower resource usage
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HUGGING_FACE_API_KEY
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=HUGGING_FACE_API_KEY
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

def get_answer(question, context):
    try:
        # Prepare the prompt
        prompt = f"""Answer the question based on the given context.

Context: {context}

Question: {question}

Answer:"""

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=2048,
            num_beams=6,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    except Exception as e:
        logger.error(f"Error in get_answer: {str(e)}")
        return "Sorry, I couldn't generate an answer."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Store extracted text in a dictionary with session IDs
pdf_texts = {}

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from PDF
        with open(filepath, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)

        # Clean up the file after processing
        os.remove(filepath)

        pdf_text = "\n".join(extracted_text)
        if not pdf_text.strip():
            return jsonify({"error": "Failed to extract text from PDF"}), 400

        # Store the text with a session ID
        session_id = "default"
        pdf_texts[session_id] = pdf_text

        return jsonify({
            "message": "PDF uploaded and processed successfully",
            "text_length": len(pdf_text)
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": "Failed to process PDF"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        session_id = "default"
        pdf_text = pdf_texts.get(session_id)
        if not pdf_text:
            return jsonify({"error": "No PDF uploaded yet"}), 400

        # Split text into chunks if it's too long
        max_length = 1024
        text_chunk = pdf_text[:max_length]

        # Get answer using the model
        answer = get_answer(question, text_chunk)

        return jsonify({
            "answer": answer,
            "model_used": "FLAN-T5-Base"
        })

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "Failed to process question"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
