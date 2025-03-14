import unittest
import json
from flask import Flask
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask_app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        """Set up test client before each test"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_endpoint(self):
        """Test the home endpoint"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_text_endpoint(self):
        """Test the text analysis endpoint"""
        test_cases = [
            {
                "text": "Wheat plants showing severe rust infection",
                "expected_terms": ["wheat", "rust"]
            },
            {
                "text": "Healthy corn growth with no issues",
                "expected_terms": ["corn"]
            }
        ]
        
        for case in test_cases:
            response = self.app.post('/analyze', 
                                   data=json.dumps({"text": case["text"]}),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            # Check response structure
            self.assertIn("agricultural_terms", data)
            self.assertIn("classification", data)
            self.assertIn("dependencies", data)
            
            # Verify expected terms are found
            found_terms = [term["text"].lower() for term in data["agricultural_terms"]]
            for expected_term in case["expected_terms"]:
                self.assertIn(
                    expected_term.lower(),
                    found_terms,
                    f"Term '{expected_term}' not found in response"
                )
    
    def test_error_handling(self):
        """Test error handling in the API"""
        # Test missing text field
        response = self.app.post('/analyze',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test empty text
        response = self.app.post('/analyze',
                               data=json.dumps({"text": ""}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test invalid JSON
        response = self.app.post('/analyze',
                               data="invalid json",
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_large_text_handling(self):
        """Test handling of large text inputs"""
        large_text = "wheat plants " * 1000
        response = self.app.post('/analyze',
                               data=json.dumps({"text": large_text}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("agricultural_terms", data)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "Wheat plants!@#$%^&*() with rust infection"
        response = self.app.post('/analyze',
                               data=json.dumps({"text": special_text}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify that agricultural terms are still identified
        found_terms = [term["text"].lower() for term in data["agricultural_terms"]]
        self.assertIn("wheat", found_terms)
        self.assertIn("rust", found_terms)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = self.app.post('/analyze',
                                   data=json.dumps({"text": "Wheat with rust"}),
                                   content_type='application/json')
            results.put(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all responses were successful
        while not results.empty():
            self.assertEqual(results.get(), 200)

if __name__ == '__main__':
    unittest.main(verbosity=2) 