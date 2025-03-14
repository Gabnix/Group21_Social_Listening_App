import spacy
import json
from spacy.training import Example
from tqdm import tqdm
from spacy.cli import download

def ensure_model_downloaded():
    """Ensure the required spaCy model is downloaded"""
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading required model...")
        download("en_core_web_sm")

def load_data_from_file(file_path, is_json=True):
    """
    Load data from a file. Can handle both JSON and plain text files.
    For JSON: Expected format is categories as keys and lists of texts as values
    For text: Returns the raw text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if is_json:
                data = json.load(file)
            else:
                data = file.read()
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not in valid JSON format.")
        return None

def preprocess_text(text, nlp):
    """
    Preprocess text by removing stopwords and punctuation
    Returns both the original token count and processed tokens
    """
    doc = nlp(text)
    token_count_before = len(doc)
    
    # Remove stopwords and punctuation
    processed_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
    token_count_after = len(processed_tokens)
    
    return {
        'original_count': token_count_before,
        'processed_count': token_count_after,
        'processed_tokens': processed_tokens
    }

def prepare_training_data(data):
    """Convert categorization data into training examples"""
    train_data = []
    for category, texts in data.items():
        for text in texts:
            cats = {cat: 1.0 if cat == category else 0.0 for cat in data.keys()}
            train_data.append((text, {"cats": cats}))
    return train_data

def train_categorizer(nlp, train_data, n_iter=20):
    """Train the text categorizer"""
    # Add text categorizer to the pipeline if not present
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
        
        # Add categories
        categories = set(cat for _, annot in train_data for cat in annot['cats'].keys())
        for category in categories:
            textcat.add_label(category)
    
    # Convert training data to spaCy's format
    train_examples = []
    for text, annotations in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    
    # Initialize the model
    optimizer = nlp.initialize()
    
    # Train the model
    print("Training the categorizer...")
    with tqdm(total=n_iter) as pbar:
        for _ in range(n_iter):
            losses = {}
            nlp.update(train_examples, sgd=optimizer, losses=losses)
            pbar.update(1)
            pbar.set_description(f"Loss: {losses['textcat']:.3f}")

def save_processed_results(output_file, preprocessing_results, categorization_results=None):
    """Save preprocessing and optional categorization results to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Preprocessing Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original token count: {preprocessing_results['original_count']}\n")
        f.write(f"Processed token count: {preprocessing_results['processed_count']}\n")
        f.write("\nProcessed tokens:\n")
        f.write("-" * 20 + "\n")
        for token in preprocessing_results['processed_tokens']:
            f.write(f"{token.text}\n")
        
        if categorization_results:
            f.write("\nCategorization Results:\n")
            f.write("-" * 20 + "\n")
            for category, score in categorization_results.items():
                f.write(f"{category}: {score:.2f}\n")

def main():
    # Ensure model is downloaded
    ensure_model_downloaded()
    
    # Create blank English model and load pretrained components
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "attribute_ruler", "matcher"])
    
    # Load and preprocess text data
    text_data = load_data_from_file('robotics_data.txt', is_json=False)
    if text_data:
        preprocessing_results = preprocess_text(text_data, nlp)
        
        # Load categorization training data
        categories_data = load_data_from_file('categories_data.json', is_json=True)
        if categories_data:
            # Prepare and train categorizer
            train_data = prepare_training_data(categories_data)
            train_categorizer(nlp, train_data)
            
            # Categorize the preprocessed text
            doc = nlp(text_data)
            categorization_results = doc.cats
            
            # Save all results
            save_processed_results('processed_results.txt', 
                                preprocessing_results, 
                                categorization_results)
        else:
            # Save only preprocessing results if categorization data is not available
            save_processed_results('processed_results.txt', 
                                preprocessing_results)

if __name__ == "__main__":
    main() 