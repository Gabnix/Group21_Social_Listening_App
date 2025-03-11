from flask import Flask, request, jsonify
import spacy
from werkzeug.exceptions import BadRequest
from SpaCy.combined_analyzer import CombinedAnalyzer

app = Flask(__name__)

# Initialize the CombinedAnalyzer
analyzer = CombinedAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": analyzer.nlp.meta['name']
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Process text with spaCy and return analysis results."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    # Use CombinedAnalyzer to process the text
    preprocessing_results, classification_results, dependency_results = analyzer.analyze_text(text)
    
    # Convert spaCy tokens to serializable format
    processed_tokens = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_
        } for token in preprocessing_results['processed_tokens']
    ]
    
    # Prepare the response
    result = {
        "preprocessing": {
            "original_count": preprocessing_results['original_count'],
            "processed_count": preprocessing_results['processed_count'],
            "processed_text": preprocessing_results['processed_text'],
            "processed_tokens": processed_tokens
        },
        "classification": classification_results,
        "dependencies": dependency_results
    }
    
    return jsonify(result)

@app.route('/preprocess', methods=['POST'])
def preprocess_text():
    """Preprocess text using CombinedAnalyzer."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    preprocessing_results = analyzer.preprocess_text(text)
    
    # Convert spaCy tokens to serializable format
    processed_tokens = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_
        } for token in preprocessing_results['processed_tokens']
    ]
    
    return jsonify({
        "original_count": preprocessing_results['original_count'],
        "processed_count": preprocessing_results['processed_count'],
        "processed_text": preprocessing_results['processed_text'],
        "processed_tokens": processed_tokens
    })

@app.route('/dependencies', methods=['POST'])
def analyze_dependencies():
    """Analyze dependencies in text using CombinedAnalyzer."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    dependencies = analyzer.analyze_dependencies(text)
    
    return jsonify({"dependencies": dependencies})

@app.route('/train', methods=['POST'])
def train_classifier():
    """Train the text classifier with provided data."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'training_data' not in data:
        return jsonify({"error": "No training data provided"}), 400
    
    try:
        training_data = analyzer.prepare_training_data(data['training_data'])
        analyzer.train_classifier(training_data, n_iter=data.get('n_iter', 20))
        return jsonify({"message": "Classifier trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/classify', methods=['POST'])
def classify_text():
    """Classify text using the trained classifier."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    classification = analyzer.classify_text(text)
    
    return jsonify({"classification": classification})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)