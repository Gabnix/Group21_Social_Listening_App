from flask import Flask, request, jsonify
import spacy
from werkzeug.exceptions import BadRequest
from SpaCy.combined_analyzer import CombinedAnalyzer
import json
import os

app = Flask(__name__)

# Initialize the CombinedAnalyzer
analyzer = CombinedAnalyzer()

# Load and train with output.json at startup
def load_and_train():
    json_path = os.path.join('SpaCy', 'output.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'train_data' in data:
                # Transform data into the format expected by CombinedAnalyzer
                training_examples = []
                categories = set()
                
                # First pass: collect all unique categories
                for example in data['train_data']:
                    for cat, value in example['cats'].items():
                        categories.add(cat)
                
                # Second pass: create training examples
                for example in data['train_data']:
                    text = example['text']
                    cats = {cat: example['cats'].get(cat, 0.0) for cat in categories}
                    training_examples.append((text, {"cats": cats}))
                
                print(f"Loaded {len(training_examples)} training examples")
                print(f"Categories found: {', '.join(categories)}")
                
                # Train the classifier with improved parameters
                analyzer.train_classifier(
                    training_examples,
                    n_iter=30,
                    dropout=0.2,
                    batch_size=8
                )
                print("Classifier trained successfully")
                return True
    except Exception as e:
        print(f"Error loading or training with output.json: {str(e)}")
        return False

# Load and train at startup
load_and_train()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Agricultural Text Analysis API",
        "endpoints": {
            "/": "Home (GET)",
            "/health": "Health check (GET)",
            "/analyze": "Analyze text (POST)",
            "/preprocess": "Preprocess text (POST)",
            "/dependencies": "Analyze dependencies (POST)",
            "/classify": "Classify text (POST)",
            "/train": "Train classifier (POST)",
            "/retrain": "Retrain classifier (POST)",
            "/test": "Test classifier (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": analyzer.nlp.meta['name'],
        "classifier_trained": "textcat" in analyzer.nlp.pipe_names
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Process text with spaCy and return analysis results."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text','']
    
    # Use CombinedAnalyzer to process the text
    preprocessing_results, classification_results, dependency_results, sentiment_result = analyzer.analyze_text(text)
    
    # Create the response with the correct structure
    result = {
        "preprocessing": {
            "processed_text": preprocessing_results["processed_text"],
            "agricultural_terms": preprocessing_results["agricultural_terms"]
        },
        "classification": classification_results,
        "dependencies": dependency_results,
        "sentiment score": sentiment_result
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

@app.route('/retrain', methods=['POST'])
def retrain_classifier():
    """Retrain the classifier using output.json."""
    try:
        success = load_and_train()
        if success:
            return jsonify({"message": "Classifier retrained successfully"})
        else:
            return jsonify({"error": "Failed to retrain classifier"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test_classifier():
    """Test the classifier with new examples."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    # Get classification results
    classification = analyzer.classify_text(text)
    
    # Sort categories by confidence score
    sorted_categories = sorted(classification.items(), key=lambda x: x[1], reverse=True)
    
    return jsonify({
        "text": text,
        "classifications": dict(sorted_categories),
        "top_category": sorted_categories[0][0],
        "confidence": sorted_categories[0][1]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)