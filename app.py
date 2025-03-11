from flask import Flask, request, jsonify
import spacy
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Load the spaCy model
# You can change this to other models like 'en_core_web_lg' for more features
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Model not found. Please download it with: python -m spacy download en_core_web_sm")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "model": nlp.meta['name']})

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Process text with spaCy and return analysis results."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Prepare the response
    result = {
        "tokens": [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct
            } for token in doc
        ],
        "entities": [
            {
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_
            } for ent in doc.ents
        ],
        "sentences": [str(sent) for sent in doc.sents]
    }
    
    return jsonify(result)

@app.route('/similarity', methods=['POST'])
def compare_texts():
    """Compare similarity between two texts."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text1' not in data or 'text2' not in data:
        return jsonify({"error": "Both text1 and text2 are required"}), 400
    
    text1 = data['text1']
    text2 = data['text2']
    
    # Process both texts
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Calculate similarity
    similarity = doc1.similarity(doc2)
    
    return jsonify({
        "text1": text1,
        "text2": text2,
        "similarity": similarity
    })

@app.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    """Extract keywords from text based on POS tags."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    # Process the text
    doc = nlp(text)
    
    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop]
    
    return jsonify({
        "text": text,
        "keywords": keywords
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)