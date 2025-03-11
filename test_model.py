import json
import requests
from pprint import pprint

def test_classifier():
    # Load test cases
    with open('test_cases.json', 'r') as f:
        test_data = json.load(f)

    # Test each case
    results = []
    for case in test_data['test_cases']:
        # Send request to the classifier
        response = requests.post(
            'http://localhost:5000/test',
            json={'text': case['text']}
        )
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                'text': case['text'],
                'expected': case['expected_category'],
                'predicted': result['top_category'],
                'confidence': result['confidence'],
                'all_categories': result['classifications']
            })
        else:
            print(f"Error testing case: {case['text']}")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")

    # Print results
    print("\nClassification Results:")
    print("=" * 80)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Expected Category: {result['expected']}")
        print(f"Predicted Category: {result['predicted']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nTop 3 Categories:")
        # Get top 3 categories by confidence
        sorted_cats = sorted(result['all_categories'].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, conf in sorted_cats:
            print(f"  {cat}: {conf:.2%}")
        print("-" * 80)

if __name__ == "__main__":
    test_classifier() 