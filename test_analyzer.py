from preprocessing.combined_analyzer import CombinedAnalyzer
import json

def test_analyzer():
    # Initialize the analyzer
    print("Initializing analyzer...")
    analyzer = CombinedAnalyzer()
    
    # Test cases
    test_cases = [
        "Wheat plants in field A showing severe rust infection on leaves",
        "Barley crop affected by powdery mildew with yellowing leaves",
        "Healthy corn plants with good growth and no visible symptoms"
    ]
    
    print("\nAnalyzing test cases:")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\nInput text: {text}")
        
        # Get the tokenized sentence
        doc = analyzer.nlp(text)
        print("\nTokenized sentence:")
        for token in doc:
            print(f"Token: {token.text:<15} Lemma: {token.lemma_:<15} POS: {token.pos_:<10} Tag: {token.tag_:<10}")
        
        # Get full analysis
        preprocessing, classification, dependencies = analyzer.analyze_text(text)
        
        print("\nPreprocessing Results:")
        print(f"Processed text: {preprocessing['processed_text']}")
        
        print("\nIdentified agricultural terms:")
        for term in preprocessing["agricultural_terms"]:
            print(f"- {term['text']} ({term['label']}) at position {term['start']}:{term['end']}")
        
        if classification:
            print("\nClassification results:")
            print(f"Category: {classification['classification']['category']}")
            print("Confidence scores:")
            for category, score in classification['classification']['scores'].items():
                print(f"- {category}: {score:.2f}")
        
        print("\nDependency analysis:")
        for dep in dependencies:
            children_str = f", children: [{', '.join(dep['children'])}]" if dep['children'] else ""
            print(f"- {dep['token']} ({dep['pos']}) -> {dep['head']} [{dep['dependency']}]{children_str}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_analyzer() 