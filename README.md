# IILM Admission Chatbot

An intelligent conversational chatbot designed to assist prospective students with admission-related queries at the Indian Institute of Leadership & Management (IILM). The chatbot provides information about admission processes, course requirements, documentation, and other relevant details.

## Features

- **Intelligent Intent Recognition**: Uses machine learning (TF-IDF vectorization + Logistic Regression) to understand user queries
- **Multi-Intent Support**: Handles various admission-related topics including:
  - Admission processes for different programs
  - Course information (BA LLB, B.Tech, MBA, BBA, B.Des, BSc, etc.)
  - Required documentation
  - Step-by-step admission procedures
  - Frequently asked questions
- **Conversation State Management**: Maintains context across conversations for better user experience
- **Confidence Scoring**: Implements multiple thresholds for accurate response matching
- **Web Interface**: User-friendly Flask-based web application
- **Response Suggestions**: Provides follow-up suggestions to guide users

## Project Structure

```
├── app.py                          # Flask web application
├── chatbot.py                      # Core chatbot logic and response generation
├── train.py                        # Model training script
├── chatbot_brain.py               # Advanced intent matching logic
├── nlp_utils.py                   # Text preprocessing utilities
├── evaluate_random_100.py         # Evaluation script for testing
├── validate_intents_official.py   # Intent validation
├── build_rich_intents.py          # Intent data augmentation
├── reset_response_dataset.py      # Dataset reset utility
├── intents.json                   # Intent patterns and responses database
├── requirements.txt               # Python dependencies
├── model.pkl                      # Pre-trained ML model
├── vectorizer.pkl                 # TF-IDF vectorizer
├── static/                        # Static assets (CSS, JavaScript, images)
├── templates/
│   └── index.html                # Web interface template
└── README.md                      # This file
```

## Requirements

- Python >= 3.8
- Flask >= 3.0.0
- scikit-learn >= 1.7.0
- NumPy >= 1.26.0
- SciPy >= 1.13.0
- NLTK (for NLP preprocessing)
- JSON (built-in)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ManishChaudhary-git/AI-ML-Project.git
cd AI-ML-Project
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **macOS/Linux**: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the chatbot model with the intents data:

```bash
python train.py
```

This will generate:
- `model.pkl` - Trained Logistic Regression model
- `vectorizer.pkl` - TF-IDF vectorizer

### Running the Chatbot Web Application

Start the Flask server:

```bash
python app.py
```

The chatbot will be available at `http://127.0.0.1:5000/`

### Chatbot Capabilities

The chatbot handles various admission-related queries such as:
- "How do I apply for Computer Science at IILM?"
- "What documents are needed for admission?"
- "What is the step-by-step process for MBA admission?"
- "Tell me about BA LLB program"
- Course inquiries for: B.Tech, MBA, BBA, B.Des, BSc, CLAT, Data Science, AI/ML, Cloud Computing, UI/UX, Cybersecurity, and more

## Configuration

Key parameters in `chatbot.py`:

- `CONFIDENCE_THRESHOLD`: 0.42 - Minimum confidence for response matching
- `HIGH_SIMILARITY_THRESHOLD`: 0.56 - High similarity cutoff
- `LOW_CONFIDENCE_SIMILARITY_THRESHOLD`: 0.38 - Low confidence similarity cutoff
- `BRAIN_CLASSIFIER_WEIGHT`: 0.56 - Weight for classifier score
- `BRAIN_PATTERN_WEIGHT`: 0.34 - Weight for pattern matching
- `BRAIN_KEYWORD_WEIGHT`: 0.18 - Weight for keyword matching

## API Endpoints

### POST `/get`
Send a user message and receive a chatbot response.

**Request:**
```json
{
  "message": "How do I apply for Computer Science?"
}
```

**Response:**
```json
{
  "response": "To apply for Computer Science at IILM...",
  "suggestions": ["What are the fees?", "What documents do I need?"]
}
```

## Natural Language Processing

The chatbot uses several NLP techniques:

1. **Text Preprocessing**:
   - Lowercase conversion
   - Punctuation removal
   - Stop word removal
   - Tokenization

2. **Feature Extraction**:
   - TF-IDF vectorization
   - Pattern-based matching
   - Keyword matching

3. **Intent Classification**:
   - Logistic Regression classifier
   - Cosine similarity scoring
   - Multi-weighted scoring system

## Evaluation & Testing

### Validate Intents
```bash
python validate_intents_official.py
```

### Evaluate Performance
```bash
python evaluate_random_100.py
```

### Generate Evaluation Report
Results are saved in `evaluation_100_report.json`

## Intent Database

The chatbot's knowledge base is stored in `intents.json` with the following structure:

```json
{
  "intents": [
    {
      "tag": "admission_process",
      "patterns": ["how do i apply for ...", "what is the step by step process for ..."],
      "responses": ["To apply for [program] at IILM..."],
      "context": "admission"
    }
  ]
}
```

### Supported Programs

- B.Tech (BTech, CSE, Engineering)
- MBA
- BBA
- B.Des
- B.Sc
- BA LLB
- Law
- Data Science
- AI/ML
- Cloud Computing
- Cybersecurity
- UI/UX
- Product Design
- Full Stack
- Business Administration
- Visual Communication
- Human Resources
- Marketing
- Finance
- Commerce
- Analytics
- Legal Studies

## Conversation State Management

The chatbot maintains session state to provide contextual responses:

```python
session["chat_state"] = {
    "last_intent": "admission_process",
    "program": "Computer Science",
    "context": "admission"
}
```

## Error Handling

The chatbot gracefully handles:
- Invalid input types
- Unclear queries (returns suggestions instead of errors)
- Low confidence responses
- Session timeouts

## Future Enhancements

- Integration with IILM's actual database
- Multi-language support
- Advanced NLP models (BERT, GPT-based)
- Live chat handoff to human agents
- Admission status tracking
- Document upload and verification

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Team

- **Project Lead**: Manish Chaudhary
- **Repository**: [GitHub - ManishChaudhary-git/AI-ML-Project](https://github.com/ManishChaudhary-git/AI-ML-Project)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Flask framework for web development
- Scikit-learn for machine learning
- NLTK for natural language processing
- IILM for domain expertise and content

---

**Last Updated**: April 26, 2026
