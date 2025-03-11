# NLP Text Analysis API

A Flask-based REST API that provides various Natural Language Processing (NLP) capabilities using spaCy.

## Features

- Text analysis (tokenization, POS tagging, dependency parsing)
- Named Entity Recognition (NER)
- Text similarity comparison
- Keyword extraction

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /health`
  - Returns the status of the API and the loaded spaCy model

### Text Analysis
- `POST /analyze`
  - Analyzes text and returns tokens, entities, and sentences
  - Request body: `{"text": "Your text here"}`

### Text Similarity
- `POST /similarity`
  - Compares similarity between two texts
  - Request body: `{"text1": "First text", "text2": "Second text"}`

### Keyword Extraction
- `POST /extract-keywords`
  - Extracts keywords from text
  - Request body: `{"text": "Your text here"}`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 