import spacy
import json
from spacy.training import Example
from tqdm import tqdm
from spacy.cli import download
import random
from spacy.matcher import Matcher
import numpy as np
from pathlib import Path

#For RoBERTa integration
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

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

        #Initialise RoBERTa model
        #Pretrained RoBERTa model for Sentiment Analysis 
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") 
        self.roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        #Add predefined keyword to inprove sentiment accuracy
        self.positive_keywords = {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15}
        self.negative_keywords = {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}


        #Initialise RoBERTa model
        #Pretrained RoBERTa model for Sentiment Analysis 
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") 
        self.roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        #Add predefined keyword to inprove sentiment accuracy
        self.positive_keywords = {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15}
        self.negative_keywords = {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}

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
            with open("SpaCy/categories_data.json", "r") as f:
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

    def train_classifier(self, n_iter=50):
        """Train text classifier with agricultural focus"""
        # Prepare training data
        train_texts = []
        train_labels = []
        
        for category, examples in self.categories_data.items():
            for example in examples:
                train_texts.append(example)
                train_labels.append(category)
        
        # Split into training and validation sets
        indices = list(range(len(train_texts)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        # Training configuration
        config = {
            "batch_size": 4,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "patience": 10,
            "min_loss_improvement": 0.001
        }
        
        # Initialize text classifier
        textcat = self.nlp.add_pipe("textcat", config={"exclusive_classes": True})
        
        # Add labels
        for label in set(train_labels):
            textcat.add_label(label)
        
        # Training loop
        print("Training agricultural text classifier...")
        best_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        # Disable other pipeline components during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):
            for i in range(n_iter):
                losses = {}
                random.shuffle(train_indices)
                
                # Training batch
                for batch_start in tqdm(range(0, len(train_indices), config["batch_size"])):
                    batch_end = min(batch_start + config["batch_size"], len(train_indices))
                    batch_indices = train_indices[batch_start:batch_end]
                    
                    batch_texts = [train_texts[i] for i in batch_indices]
                    batch_labels = [train_labels[i] for i in batch_indices]
                    
                    docs = [self.nlp.make_doc(text) for text in batch_texts]
                    examples = []
                    for j, doc in enumerate(docs):
                        label = batch_labels[j]
                        cats = {l: (l == label) for l in textcat.labels}
                        examples.append(Example.from_dict(doc, {"cats": cats}))
                    
                    self.nlp.update(examples, drop=config["dropout"], losses=losses)
                
                # Validation
                val_losses = []
                for val_index in val_indices:
                    val_text = train_texts[val_index]
                    val_label = train_labels[val_index]
                    
                    doc = self.nlp.make_doc(val_text)
                    cats = {l: (l == val_label) for l in textcat.labels}
                    example = Example.from_dict(doc, {"cats": cats})
                    val_losses.append(textcat.get_loss(example))
                
                val_loss = np.mean(val_losses)
                
                # Early stopping check
                if val_loss < best_loss - config["min_loss_improvement"]:
                    best_loss = val_loss
                    patience_counter = 0
                    best_weights = textcat.model.copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at iteration {i+1}")
                    break
                
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}: Train Loss: {losses['textcat']:.3f}, Val Loss: {val_loss:.3f}")
        
        # Restore best weights
        if best_weights is not None:
            textcat.model = best_weights
        
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

    
    """def analyze_text(self, text, save_path=None):
        """"""Perform comprehensive text analysis: preprocessing, classification, and dependency parsing""""""
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
            
        return preprocessing_results, classification_results, dependency_results"""
    
    #new analyse_text function
    def analyze_text(self, text, save_path=None):

        # Preprocess text
        preprocessing_results = self.preprocess_text(text)

        # Classify text if textcat is in pipeline
        classification_results = None
        if "textcat" in self.nlp.pipe_names:
            classification_results = self.classify_text(preprocessing_results['processed_text'])
        
        # Analyze dependencies
        dependency_results = self.analyze_dependencies(text)

        # RoBERTa Sentiment Analysis
        sentiment_results = self.sentiment_analysis(preprocessing_results['processed_text'])

        # Save results if path provided
        if save_path:
            self.save_results(save_path, preprocessing_results, classification_results, dependency_results, sentiment_results)
            
        return preprocessing_results, classification_results, dependency_results, sentiment_results


    def sentiment_analysis(self, text):
        """"Perform Sentiment Analysis using RoBERTa model"""
        # Tokenize the text to ensure the size = 512
        input = self.roberta_tokenizer(text, return_tensors="pt",truncation=True, padding=True)
        
        with torch.no_grad():
            # Perform forward pass
            output = self.roberta_model(**input)

        # Get the predicted sentiment
        sentiment = torch.softmax(output.logits, dim=-1)
        score = sentiment[0].tolist()

        # Get the enhance the sentiment score
        new_score = self.sentiment_score_refinement(text, score)
        new_score = [round(score, 3) for score in new_score]

        formatted_score = {
            "Negative": new_score[0],
            "Neutral": new_score[1],
            "Positive": new_score[2]
        }
        return formatted_score

    def sentiment_score_refinement(self, text, sentiment_score):
        """Refine the score using predefined keywords"""
        tune_score = 0
        newText = text.lower()

        # Check for positive keywords
        for keyword, adjustment in self.positive_keywords.items():
            if keyword in newText:
                tune_score += adjustment

        # Check for negative keywords
        for keyword, adjustment in self.negative_keywords.items():
            if keyword in newText:
                tune_score -= abs(adjustment)

        # Update the sentiment score
        # Apply positive adjustments
        sentiment_score[2] = min(1.0, max(0.0, sentiment_score[2] + tune_score)) #to ensure the score is between 0 and 1
        # Apply negative adjustments
        sentiment_score[0] = min(1.0, max(0.0, sentiment_score[0] - tune_score)) 
        
        return sentiment_score

    def save_results(self, text, classification_results, dependency_results, sentiment_results, output_file="analysis_results.json"):
        """Save analysis results to file"""
        output = {
            "input_text": text,
            "classification": classification_results["classification"],
            "agricultural_terms": classification_results["agricultural_terms"],
            "dependencies": dependency_results,
            "sentiment score": sentiment_results
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
        preprocessing_results, classification_results, dependency_results, sentiment_results = analyzer.analyze_text(
            text_data, 
            save_path='combined_analysis_results.json'
        )
        
        print("\nAnalysis completed! Results have been saved to 'combined_analysis_results.json'")

if __name__ == "__main__":
    main() 