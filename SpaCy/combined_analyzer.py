import spacy
import json
from spacy.training import Example
from tqdm import tqdm
from spacy.cli import download
import random
from spacy.matcher import Matcher

#For RoBERTa integration
import torch
import numpy as np
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
        
    def add_matcher_patterns(self):
        """Add patterns to the matcher for identifying agricultural terms and diseases"""
        # Disease patterns
        disease_pattern = [
            [{"LOWER": {"IN": ["rust", "mildew", "blight", "rot", "spot", "wilt"]}},
             {"OP": "?", "LOWER": {"IN": ["disease", "infection", "infestation"]}}],
            [{"LOWER": {"IN": ["leaf", "stem", "root"]}},
             {"LOWER": {"IN": ["rust", "rot", "spot", "disease"]}}]
        ]
        
        # Crop patterns
        crop_pattern = [
            [{"LOWER": {"IN": ["wheat", "barley", "corn", "rice", "soybean", "potato"]}}],
            [{"LOWER": {"IN": ["crop", "plant", "field"]}},
             {"LOWER": "health"}]
        ]
        
        # Symptom patterns
        symptom_pattern = [
            [{"LOWER": {"IN": ["yellow", "brown", "black", "white"]}},
             {"LOWER": {"IN": ["spots", "lesions", "patches", "streaks"]}}],
            [{"LOWER": {"IN": ["wilting", "stunted", "discolored", "infected"]}}]
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

    def train_classifier(self, training_data, n_iter=50, dropout=0.1, batch_size=4):
        """Train the text classifier with improved parameters
        
        Args:
            training_data: List of (text, annotations) tuples
            n_iter: Number of training iterations (increased for better convergence)
            dropout: Dropout rate (reduced to allow better initial learning)
            batch_size: Size of batches for training (reduced for more frequent updates)
        """
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
        
        # Split training data into train and validation sets (80/20 split)
        random.shuffle(train_examples)  # Shuffle before splitting
        split = int(len(train_examples) * 0.8)
        train_data = train_examples[:split]
        dev_data = train_examples[split:]
        
        # Initialize optimizer with component-specific learning rates
        optimizer = self.nlp.initialize()
        
        # Early stopping configuration
        patience = 10  # Increased patience for more chances to improve
        best_loss = float('inf')
        no_improvement = 0
        min_loss_improvement = 0.001  # Minimum improvement threshold
        best_weights = None
        
        print("Training the classifier...")
        with tqdm(total=n_iter) as pbar:
            for i in range(n_iter):
                # Shuffle training data each iteration
                random.shuffle(train_data)
                
                # Create smaller batches for more frequent updates
                batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
                
                # Training step
                losses = {}
                for batch in batches:
                    self.nlp.update(
                        batch,
                        sgd=optimizer,
                        drop=dropout,
                        losses=losses
                    )
                
                # Calculate validation loss
                val_loss = 0.0
                if dev_data:
                    val_losses = {}
                    # Evaluate on validation set
                    for example in dev_data:
                        scores = self.nlp(example.text).cats
                        for label, gold_score in example.y.cats.items():
                            pred_score = scores.get(label, 0.0)
                            val_loss += (gold_score - pred_score) ** 2
                    val_loss /= len(dev_data)
                
                    # Early stopping with minimum improvement threshold
                    if val_loss < best_loss - min_loss_improvement:
                        best_loss = val_loss
                        no_improvement = 0
                        # Store the current model weights
                        best_weights = self.nlp.to_bytes()
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            print(f"\nStopping early at iteration {i+1} due to no significant improvement")
                            # Restore best weights
                            if best_weights is not None:
                                self.nlp.from_bytes(best_weights)
                            break
                
                # Update progress bar with both losses
                train_loss = losses.get('textcat', 0.0)
                pbar.update(1)
                pbar.set_description(f"Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
        
        print(f"\nFinal training loss: {train_loss:.3f}")
        print(f"Final validation loss: {val_loss:.3f}")
        print("Classifier trained successfully")
        
        return train_loss, val_loss

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
            self.save_results(save_path, preprocessing_results, classification_results, dependency_results)
            
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
        return new_score

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

    def save_results(self, output_file, preprocessing_results, classification_results=None, dependency_results=None, sentiment_results=None):
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

            if sentiment_results:
                f.write("\nSentiment Analysis Results:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Negative: {sentiment_results[0]}\n")
                f.write(f"Neutral: {sentiment_results[1]}\n")
                f.write(f"Positive: {sentiment_results[2]}\n")

def main():
    # Initialize analyzer
    analyzer = CombinedAnalyzer()
    
    # Load training data for classification
    training_data = analyzer.load_data('categories_data.json')
    if training_data:
        # Prepare and train classifier
        train_data = analyzer.prepare_training_data(training_data)
        train_loss, val_loss = analyzer.train_classifier(train_data)
    
    # Load and analyze text
    text_data = analyzer.load_data('input.txt', is_json=False)
    if text_data:
        # Perform analysis and save results
        preprocessing_results, classification_results, dependency_results, sentiment_results = analyzer.analyze_text(
            text_data, 
            save_path='combined_analysis_results.txt'
        )
        
        print("\nAnalysis completed! Results have been saved to 'combined_analysis_results.txt'")

if __name__ == "__main__":
    main() 