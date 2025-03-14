import spacy
import json
from spacy.training import Example
from tqdm import tqdm
from spacy.cli import download
import random
from spacy.matcher import Matcher
import numpy as np
from pathlib import Path
import spacy.util

class CombinedAnalyzer:
    def __init__(self):
        """Initialize the CombinedAnalyzer with required models and components"""
        self.ensure_model_downloaded()
        # Create spaCy model with all necessary components
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])  # Disable NER for faster processing
        
        # Configure pipeline components
        if "tagger" not in self.nlp.pipe_names:
            self.nlp.add_pipe("tagger", before="parser")
        
        if "attribute_ruler" not in self.nlp.pipe_names:
            self.nlp.add_pipe("attribute_ruler", after="tagger")
            
        # Configure attribute ruler patterns
        ruler = self.nlp.get_pipe("attribute_ruler")
        patterns = [
            {"patterns": [[{"ORTH": "rust"}]], "attrs": {"TAG": "NN"}},
            {"patterns": [[{"ORTH": "mildew"}]], "attrs": {"TAG": "NN"}},
            {"patterns": [[{"ORTH": "wheat"}]], "attrs": {"TAG": "NN"}},
            {"patterns": [[{"ORTH": "barley"}]], "attrs": {"TAG": "NN"}},
            {"patterns": [[{"ORTH": "disease"}]], "attrs": {"TAG": "NN"}},
            {"patterns": [[{"ORTH": "infection"}]], "attrs": {"TAG": "NN"}}
        ]
        for pattern in patterns:
            ruler.add(pattern["patterns"], pattern["attrs"])
        
        # Initialize and configure the matcher with patterns
        self.matcher = Matcher(self.nlp.vocab)
        self.add_matcher_patterns()
        
        # Print pipeline information
        print("Active pipeline components:", self.nlp.pipe_names)
        
        # Agricultural-specific stopwords
        self.agri_stopwords = {"field", "farm", "crop", "plant", "seed", "grow", "harvest"}
        
        # Load training data
        self.load_training_data()
        
    def add_matcher_patterns(self):
        """Add patterns to the matcher for identifying agricultural terms and diseases"""
        # Disease patterns
        disease_pattern = [
            [{"LOWER": {"IN": ["rust", "mildew", "smut", "blight", "rot", "spot", "mosaic", "wilt", "canker", "scab"]}}],
            [{"LOWER": "powdery"}, {"LOWER": "mildew"}],
            [{"LOWER": "leaf"}, {"LOWER": "spot"}],
            [{"LOWER": "stem"}, {"LOWER": "rust"}],
            [{"LOWER": "black"}, {"LOWER": "leg"}],
            [{"LOWER": "root"}, {"LOWER": "rot"}]
        ]
        
        # Crop patterns
        crop_pattern = [
            [{"LOWER": {"IN": ["wheat", "barley", "corn", "rice", "soybean", "cotton", "canola", "oats"]}}],
            [{"LOWER": {"IN": ["chickpea", "lentil", "lupin", "faba", "mungbean", "safflower", "sorghum"]}}]
        ]
        
        # Symptom patterns
        symptom_pattern = [
            [{"LOWER": {"IN": ["wilting", "yellowing", "spotting", "lesion", "chlorosis", "necrosis"]}}],
            [{"LOWER": "leaf"}, {"LOWER": {"IN": ["curl", "spot", "wilt", "burn"]}}],
            [{"LOWER": "stem"}, {"LOWER": {"IN": ["canker", "rot", "lesion"]}}],
            [{"LOWER": "root"}, {"LOWER": {"IN": ["rot", "damage", "lesion"]}}]
        ]
        
        # Add patterns to matcher
        self.matcher.add("DISEASE", disease_pattern)
        self.matcher.add("CROP", crop_pattern)
        self.matcher.add("SYMPTOM", symptom_pattern)
        
        print("Matcher patterns configured for diseases, crops, and symptoms")

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

    def load_training_data(self):
        """Load and prepare training data from JSON file"""
        try:
            with open("preprocessing/categories_data.json", "r") as f:
                self.categories_data = json.load(f)
            print(f"Loaded {len(self.categories_data)} categories with {sum(len(examples) for examples in self.categories_data.values())} training examples")
        except Exception as e:
            print(f"Error loading training data: {e}")
            self.categories_data = {}

    def preprocess_text(self, text):
        """Preprocess text with agricultural domain focus"""
        # Basic preprocessing
        text = text.lower().strip()
        
        # Create SpaCy doc
        doc = self.nlp(text)
        
        # Extract agricultural terms
        matches = self.matcher(doc)
        agri_terms = []
        for match_id, start, end in matches:
            span = doc[start:end]
            agri_terms.append({
                "text": span.text,
                "label": self.nlp.vocab.strings[match_id],
                "start": start,
                "end": end
            })
        
        # Remove agricultural stopwords while keeping important domain terms
        tokens = [token.text for token in doc if not (token.is_stop and token.text not in self.agri_stopwords)]
        
        return {
            "processed_text": " ".join(tokens),
            "agricultural_terms": agri_terms,
            "doc": doc
        }

    def prepare_training_data(self, data):
        """Convert categorization data into training examples"""
        train_data = []
        for category, texts in data.items():
            for text in texts:
                cats = {cat: 1.0 if cat == category else 0.0 for cat in data.keys()}
                train_data.append((text, {"cats": cats}))
        return train_data

    def train_classifier(self, training_data=None, n_iter=50, dropout=0.1, batch_size=4):
        """Train text classifier with agricultural focus"""
        # Remove existing textcat if present
        if "textcat" in self.nlp.pipe_names:
            self.nlp.remove_pipe("textcat")
            
        # Add new textcat with default configuration
        textcat = self.nlp.add_pipe("textcat", last=True)
        
        # Prepare training data
        train_examples = []
        if training_data is None:
            # Use categories_data if no training data provided
            all_categories = list(self.categories_data.keys())
            for category, texts in self.categories_data.items():
                for text in texts:
                    doc = self.nlp.make_doc(text)
                    cats = {cat: 1.0 if cat == category else 0.0 for cat in all_categories}
                    example = {"text": doc, "cats": cats}
                    train_examples.append(example)
        else:
            # Use provided training data
            all_categories = set()
            for text, cats in training_data:
                all_categories.update(cats["cats"].keys())
            
            for text, cats_dict in training_data:
                doc = self.nlp.make_doc(text)
                cats = {cat: cats_dict["cats"].get(cat, 0.0) for cat in all_categories}
                example = {"text": doc, "cats": cats}
                train_examples.append(example)
        
        # Add labels to textcat
        for category in all_categories:
            textcat.add_label(category)
        
        # Split into training and validation sets
        random.shuffle(train_examples)
        split = int(len(train_examples) * 0.8)
        train_data = train_examples[:split]
        dev_data = train_examples[split:]
        
        # Initialize the model
        optimizer = self.nlp.initialize()
        
        # Training loop
        print(f"Training textcat model with {len(train_data)} examples...")
        with self.nlp.select_pipes(enable=["textcat"]):
            for i in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                
                # Batch training
                batches = spacy.util.minibatch(train_data, size=batch_size)
                for batch in batches:
                    examples = []
                    for eg in batch:
                        examples.append(Example.from_dict(eg["text"], {"cats": eg["cats"]}))
                    self.nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)
                
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i + 1}: Losses", losses)
        
        print("Training completed!")

    def classify_text(self, text):
        """Classify text with confidence scores"""
        # Preprocess text
        processed = self.preprocess_text(text)
        doc = processed["doc"]
        
        # Get classification scores
        scores = doc.cats
        
        # Enhance results with agricultural terms
        result = {
            "classification": {
                "category": max(scores.items(), key=lambda x: x[1])[0],
                "scores": scores
            },
            "agricultural_terms": processed["agricultural_terms"]
        }
        
        return result

    def analyze_dependencies(self, text):
        """Analyze syntactic dependencies with agricultural focus"""
        doc = self.nlp(text)
        
        # Extract dependencies with agricultural context
        deps = []
        for token in doc:
            # Focus on agricultural terms and their relationships
            if (token.dep_ in ["nsubj", "dobj", "pobj"] or 
                token.head.dep_ in ["nsubj", "dobj", "pobj"] or
                token.text.lower() in self.agri_stopwords):
                
                deps.append({
                    "token": token.text,
                    "pos": token.pos_,
                    "dependency": token.dep_,
                    "head": token.head.text,
                    "head_pos": token.head.pos_,
                    "children": [child.text for child in token.children]
                })
        
        return deps

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
            self.save_results(text, classification_results, dependency_results, save_path)
            
        return preprocessing_results, classification_results, dependency_results

    def save_results(self, text, classification_results, dependency_results, output_file="analysis_results.json"):
        """Save analysis results to file"""
        output = {
            "input_text": text,
            "classification": classification_results["classification"],
            "agricultural_terms": classification_results["agricultural_terms"],
            "dependencies": dependency_results
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"Results saved to {output_file}")

def main():
    # Initialize analyzer
    analyzer = CombinedAnalyzer()
    
    # Load training data for classification
    training_data = analyzer.load_data('categories_data.json')
    if training_data:
        # Prepare and train classifier
        train_data = analyzer.prepare_training_data(training_data)
        analyzer.train_classifier()
    
    # Load and analyze text
    text_data = analyzer.load_data('input.txt', is_json=False)
    if text_data:
        # Perform analysis and save results
        preprocessing_results, classification_results, dependency_results = analyzer.analyze_text(
            text_data, 
            save_path='combined_analysis_results.json'
        )
        
        print("\nAnalysis completed! Results have been saved to 'combined_analysis_results.json'")

if __name__ == "__main__":
    main() 