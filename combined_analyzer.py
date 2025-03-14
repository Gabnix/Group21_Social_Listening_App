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

        # Load sentiment keywords from JSON file
        self.positive_keywords, self.negative_keywords = self.load_sentiment_keywords()

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

    def load_sentiment_keywords(self):
        """Load sentiment keywords from JSON file using `with open`"""
        keywords_file = "preprocessing/roberta_sentiment_keywords.json"
        try:
            with open(keywords_file, 'r', encoding='utf-8') as file:
                keywords_data = json.load(file)
                if not keywords_data:  # Handle empty file
                    print(f"Error: File {keywords_file} is empty.")
                    return (
                        {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15},
                        {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}        
                    )
                return (
                    keywords_data.get("positive_keywords", {}),
                    keywords_data.get("negative_keywords", {})
                )
        except FileNotFoundError:
            print(f"Error: File {keywords_file} not found.")
            return (
                {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15},
                {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}        
            )
        except json.JSONDecodeError:
            print(f"Error: File {keywords_file} is not in valid JSON format.")
            return (
                {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15},
                {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}        
            )
        except Exception as e:
            print(f"Error loading sentiment keywords: {str(e)}")
            return (
                {"healthy": 0.25, "disease-free": 0.2, "thriving":0.15, "robust":0.15, "vigorous": 0.15},
                {"infected": -0.2, "diseased": -0.2, "unhealthy": -0.15, "sickly": -0.2, "weak": 0.10, "outbreak": -0.3, "pest infestation": -0.3}        
            )


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
        try:
            # Preprocess text
            processed = self.preprocess_text(text)
            doc = processed["doc"]
            
            # Get classification scores
            if "textcat" not in self.nlp.pipe_names:
                return {
                    "classification": {
                        "category": "unknown",
                        "scores": {}
                    },
                    "agricultural_terms": processed["agricultural_terms"]
                }
            
            scores = doc.cats
            
            # Enhance results with agricultural terms
            result = {
                "classification": {
                    "category": max(scores.items(), key=lambda x: x[1])[0] if scores else "unknown",
                    "scores": scores
                },
                "agricultural_terms": processed["agricultural_terms"]
            }
            
            return result
        except Exception as e:
            print(f"Classification failed: {str(e)}")
            return {
                "classification": {
                    "category": "unknown",
                    "scores": {}
                },
                "agricultural_terms": []
            }

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
        try:
            # Preprocess text
            preprocessing_results = self.preprocess_text(text)
            
            # Classify text if textcat is in pipeline
            classification_results = None
            if "textcat" in self.nlp.pipe_names:
                try:
                    classification_results = self.classify_text(preprocessing_results['processed_text'])
                except Exception as e:
                    print(f"Classification failed: {str(e)}")
                    classification_results = None
            
            # Analyze dependencies
            try:
                dependency_results = self.analyze_dependencies(text)
            except Exception as e:
                print(f"Dependency analysis failed: {str(e)}")
                dependency_results = []

            # RoBERTa Sentiment Analysis
            try:
                sentiment_results = self.sentiment_analysis(preprocessing_results['processed_text'])
            except Exception as e:
                print(f"Sentiment analysis failed: {str(e)}")
                sentiment_results = None
            
            # Save results if path provided
            if save_path:
                try:
                    self.save_results(text, classification_results, dependency_results, sentiment_results, save_path)
                except Exception as e:
                    print(f"Failed to save results: {str(e)}")
            
            return preprocessing_results, classification_results, dependency_results, sentiment_results
        except Exception as e:
            print(f"Text analysis failed: {str(e)}")
            return {}, None, [], None

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

        # Assuming new_score is a list or tuple with three values: [negative_score, neutral_score, positive_score]
        formatted_score = {
            "Negative": new_score[0],
            "Neutral": new_score[1],
            "Positive": new_score[2]
        }
        
        if new_score[0] > new_score[1] and new_score[0] > new_score[2]:
            sentiment = "Negative"
        elif new_score[1] > new_score[0] and new_score[1] > new_score[2]:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        # Add the sentiment to the formatted_score
        formatted_score["Sentiment"] = sentiment

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
            "classification": classification_results["classification"] if classification_results else None,
            "agricultural_terms": classification_results["agricultural_terms"] if classification_results else [],
            "dependencies": dependency_results if dependency_results else [],
            "sentiment_score": sentiment_results if sentiment_results else None
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