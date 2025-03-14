import unittest
import coverage
import sys
import os

def run_tests_with_coverage():
    # Start code coverage measurement
    cov = coverage.Coverage(
        branch=True,
        source=['SpaCy', 'flask_server.py'],
        omit=['tests/*', 'venv/*']
    )
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(tests_dir)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Generate coverage report
    print("\nCoverage Report:")
    print("-" * 70)
    cov.report()
    
    # Generate HTML report
    cov.html_report(directory='coverage_report')
    print("\nDetailed HTML coverage report generated in 'coverage_report' directory")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests_with_coverage()
    sys.exit(0 if success else 1) 