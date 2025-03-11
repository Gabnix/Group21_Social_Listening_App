import spacy
import json
from spacy.training import Example
from tqdm import tqdm
from spacy.cli import download

class CombinedAnalyzer:
    def __init__(self):
        """Initialize the CombinedAnalyzer with required models and components"""
        self.ensure_model_downloaded()
        # Create spaCy model with all necessary components
        self.nlp = spacy.load("en_core_web_sm")
        
    def ensure_model_downloaded(self):
        """Ensure the required spaCy model is downloaded"""
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading required model...")
            download("en_core_web_sm")

    def load_data(self, file_path, is_json=True):
        """Load data from a file (JSON or plain text)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                if is_json:
                    return json.load(file)
                return file.read()
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: File {file_path} is not in valid JSON format.")
            return None

    def preprocess_text(self, text):
        """Preprocess text by removing stopwords and punctuation"""
        doc = self.nlp(text)
        token_count_before = len(doc)
        
        # Remove stopwords and punctuation
        processed_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        token_count_after = len(processed_tokens)
        
        return {
            'original_count': token_count_before,
            'processed_count': token_count_after,
            'processed_tokens': processed_tokens,
            'processed_text': ' '.join([token.text for token in processed_tokens])
        }

    def prepare_training_data(self, data):
        """Convert categorization data into training examples"""
        train_data = []
        for category, texts in data.items():
            for text in texts:
                cats = {cat: 1.0 if cat == category else 0.0 for cat in data.keys()}
                train_data.append((text, {"cats": cats}))
        return train_data

    def train_classifier(self, training_data, n_iter=20):
        """Train the text classifier"""
        # Add text categorizer to the pipeline if not present
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.add_pipe("textcat", last=True)
            
            # Add categories
            categories = set(cat for _, annot in training_data for cat in annot['cats'].keys())
            for category in categories:
                textcat.add_label(category)
        
        # Convert training data to spaCy's format
        train_examples = []
        for text, annotations in training_data:
            train_examples.append(Example.from_dict(self.nlp.make_doc(text), annotations))
        
        # Initialize the model
        optimizer = self.nlp.initialize()
        
        # Train the model
        print("Training the classifier...")
        with tqdm(total=n_iter) as pbar:
            for _ in range(n_iter):
                losses = {}
                self.nlp.update(train_examples, sgd=optimizer, losses=losses)
                pbar.update(1)
                pbar.set_description(f"Loss: {losses['textcat']:.3f}")

    def classify_text(self, text):
        """Classify the given text"""
        doc = self.nlp(text)
        return doc.cats

    def analyze_dependencies(self, text):
        """Analyze dependencies in the text"""
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'token': token.text,
                'head': token.head.text,
                'dependency': token.dep_,
                'pos': token.pos_
            })
        
        return dependencies

    def analyze_text(self, text, save_path=None):
        """Perform comprehensive text analysis: preprocessing, classification, and dependency parsing"""
        # Preprocess text
        preprocessing_results = self.preprocess_text(text)
        
        # Classify text if textcat is in pipeline
        classification_results = None
        if "textcat" in self.nlp.pipe_names:
            classification_results = self.classify_text(preprocessing_results['processed_text'])
        
        # Analyze dependencies
        dependency_results = self.analyze_dependencies(text)
        
        # Save results if path provided
        if save_path:
            self.save_results(save_path, preprocessing_results, classification_results, dependency_results)
            
        return preprocessing_results, classification_results, dependency_results

    def save_results(self, output_file, preprocessing_results, classification_results=None, dependency_results=None):
        """Save analysis results to a file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Comprehensive Text Analysis Results\n")
            f.write("================================\n\n")
            
            f.write("Preprocessing Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original token count: {preprocessing_results['original_count']}\n")
            f.write(f"Processed token count: {preprocessing_results['processed_count']}\n")
            f.write("\nProcessed tokens:\n")
            f.write("-" * 20 + "\n")
            for token in preprocessing_results['processed_tokens']:
                f.write(f"{token.text}\n")
            
            if classification_results:
                f.write("\nClassification Results:\n")
                f.write("-" * 20 + "\n")
                for category, score in classification_results.items():
                    f.write(f"{category}: {score:.2f}\n")
            
            if dependency_results:
                f.write("\nDependency Analysis Results:\n")
                f.write("-" * 20 + "\n")
                for dep in dependency_results:
                    f.write(f"{dep['token']} -> {dep['head']} ({dep['dependency']}) [POS: {dep['pos']}]\n")

def main():
    # Initialize analyzer
    analyzer = CombinedAnalyzer()
    
    # Load training data for classification
    training_data = analyzer.load_data('categories_data.json')
    if training_data:
        # Prepare and train classifier
        train_data = analyzer.prepare_training_data(training_data)
        analyzer.train_classifier(train_data)
    
    # Load and analyze text
    text_data = analyzer.load_data('input.txt', is_json=False)
    if text_data:
        # Perform analysis and save results
        preprocessing_results, classification_results, dependency_results = analyzer.analyze_text(
            text_data, 
            save_path='combined_analysis_results.txt'
        )
        
        print("\nAnalysis completed! Results have been saved to 'combined_analysis_results.txt'")

if __name__ == "__main__":
    main() 