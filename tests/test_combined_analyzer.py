import unittest
from preprocessing.combined_analyzer import CombinedAnalyzer
import json
import os

class TestCombinedAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all tests"""
        cls.analyzer = CombinedAnalyzer()
        
    def test_initialization(self):
        """Test if the analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer.nlp)
        self.assertIsNotNone(self.analyzer.matcher)
        self.assertTrue(hasattr(self.analyzer, 'categories_data'))
        
    def test_preprocessing(self):
        """Test text preprocessing functionality"""
        test_cases = [
            {
                "input": "Wheat plants showing severe rust infection",
                "expected_terms": [
                    {"label": "CROP", "text": "wheat"},
                    {"label": "DISEASE", "text": "rust"}
                ]
            },
            {
                "input": "Healthy corn with no diseases",
                "expected_terms": [
                    {"label": "CROP", "text": "corn"}
                ]
            },
            {
                "input": "Multiple symptoms: yellowing leaves and powdery mildew",
                "expected_terms": [
                    {"label": "SYMPTOM", "text": "yellowing"},
                    {"label": "DISEASE", "text": "powdery mildew"}
                ]
            }
        ]
        
        for case in test_cases:
            result = self.analyzer.preprocess_text(case["input"])
            
            # Check if agricultural terms are identified
            found_terms = {(term["text"], term["label"]) for term in result["agricultural_terms"]}
            expected_terms = {(term["text"], term["label"]) for term in case["expected_terms"]}
            
            self.assertTrue(
                expected_terms.issubset(found_terms),
                f"Missing expected terms in '{case['input']}'. Expected: {expected_terms}, Found: {found_terms}"
            )
            
    def test_dependency_analysis(self):
        """Test dependency analysis functionality"""
        text = "Wheat plants show severe rust infection"
        deps = self.analyzer.analyze_dependencies(text)
        
        # Check if basic dependency structure is correct
        self.assertTrue(any(dep["token"] == "wheat" for dep in deps))
        self.assertTrue(any(dep["token"] == "rust" for dep in deps))
        
        # Test more complex sentence
        complex_text = "The infected wheat plants in the field show multiple symptoms"
        complex_deps = self.analyzer.analyze_dependencies(complex_text)
        
        # Verify key relationships are captured
        self.assertTrue(
            any(dep["token"] == "plants" and dep["head"] == "show" for dep in complex_deps),
            "Failed to capture subject-verb relationship"
        )
    
    def test_classification(self):
        """Test text classification functionality"""
        test_cases = [
            {
                "input": "Severe rust infection on wheat leaves",
                "expected_category": "disease"
            },
            {
                "input": "Healthy crop growth with no issues",
                "expected_category": "normal"
            }
        ]
        
        for case in test_cases:
            result = self.analyzer.classify_text(case["input"])
            self.assertIn("classification", result)
            self.assertIn("category", result["classification"])
            
            # Check if classification is as expected
            self.assertEqual(
                result["classification"]["category"],
                case["expected_category"],
                f"Incorrect classification for '{case['input']}'"
            )
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test empty input
        result = self.analyzer.analyze_text("")
        self.assertIsNotNone(result)
        
        # Test input with special characters
        special_chars = "Wheat plants with rust!@#$%^&*()"
        result = self.analyzer.analyze_text(special_chars)
        self.assertIsNotNone(result)
        
        # Test very long input
        long_text = "wheat " * 1000
        result = self.analyzer.analyze_text(long_text)
        self.assertIsNotNone(result)
    
    def test_agricultural_terms(self):
        """Test specific agricultural term recognition"""
        terms_to_test = {
            "crops": ["wheat", "barley", "corn", "rice", "soybean"],
            "diseases": ["rust", "mildew", "blight", "rot"],
            "symptoms": ["wilting", "yellowing", "spotting"]
        }
        
        for category, terms in terms_to_test.items():
            for term in terms:
                result = self.analyzer.preprocess_text(f"Testing {term} recognition")
                found = any(
                    term.lower() in term_info["text"].lower() 
                    for term_info in result["agricultural_terms"]
                )
                self.assertTrue(
                    found,
                    f"Failed to recognize {category} term: {term}"
                )
    
    def test_save_results(self):
        """Test results saving functionality"""
        test_text = "Wheat plants with rust infection"
        output_file = "test_results.json"
        
        # Get analysis results
        preprocessing, classification, dependencies = self.analyzer.analyze_text(test_text)
        
        # Save results
        self.analyzer.save_results(test_text, classification, output_file)
        
        # Verify file was created and contains valid JSON
        self.assertTrue(os.path.exists(output_file))
        
        try:
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
                
            # Check if all required fields are present
            self.assertIn("input_text", saved_data)
            self.assertIn("classification", saved_data)
            self.assertIn("agricultural_terms", saved_data)
            self.assertIn("dependencies", saved_data)
        finally:
            # Clean up test file
            if os.path.exists(output_file):
                os.remove(output_file)

if __name__ == '__main__':
    unittest.main(verbosity=2) 