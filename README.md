# FUTURE_ML_02

# Support Ticket Classification & Prioritization System - Machine Learning

An intelligent machine learning system that automatically classifies support tickets into categories and assigns priority levels to help support teams respond faster and more efficiently.

## 🎯 Overview

This system is designed for real-world use in SaaS companies, help desks, and IT support operations. It uses Natural Language Processing (NLP) and Machine Learning to:

1. **Classify support tickets** into 8 distinct categories
2. **Assign priority levels** (High/Medium/Low)
3. **Provide confidence scores** for each prediction
4. **Help support teams** respond faster and more efficiently

## 📊 Performance

- **Test Accuracy**: 84.82%
- **Training Accuracy**: 87.42%
- **Improvement over Naive Bayes baseline**: +99.59%
- **Weighted F1-Score**: 0.8486
- **Training Time**: ~3 seconds

## 🗂️ Categories

The system classifies tickets into 8 categories:

1. **Hardware** (28.47% of tickets)
2. **HR Support** (22.82%)
3. **Access** (14.89%)
4. **Miscellaneous** (14.76%)
5. **Storage** (5.81%)
6. **Purchase** (5.15%)
7. **Internal Project** (4.43%)
8. **Administrative rights** (3.68%)

## 🚀 Features

- **Automated Classification**: Instantly categorize incoming support tickets
- **Priority Assignment**: Automatically assign priority levels based on urgency
- **Confidence Scores**: Get prediction confidence for each classification
- **Comprehensive Analytics**: Detailed performance metrics and visualizations
- **Production Ready**: Optimized for real-world deployment

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
kagglehub>=0.3.0
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ticket-classification.git
cd ticket-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the notebook or Python script:
```bash
jupyter notebook Ticket_Classification.ipynb
```

## 💻 Usage

### Basic Usage

```python
import pickle

# Load the trained model
with open('final_classifier.pkl', 'rb') as f:
    model_package = pickle.load(f)

vectorizer = model_package['vectorizer']
classifier = model_package['classifier']

# Classify a new ticket
ticket = "My laptop is not turning on. This is urgent!"
X_ticket = vectorizer.transform([ticket])
category = classifier.predict(X_ticket)[0]
confidence = classifier.predict_proba(X_ticket)[0].max() * 100

print(f"Category: {category}")
print(f"Confidence: {confidence:.2f}%")
```

### Example Predictions

```python
test_tickets = [
    "My laptop is not turning on. This is urgent!",
    "I need access to the sales database",
    "Can you help me reset my password?",
    "We need to purchase 10 new monitors",
    "The printer keeps jamming"
]

for ticket in test_tickets:
    X = vectorizer.transform([ticket])
    category = classifier.predict(X)[0]
    confidence = classifier.predict_proba(X)[0].max() * 100
    print(f"Ticket: {ticket[:50]}...")
    print(f"→ {category} ({confidence:.1f}% confidence)\n")
```

## 📈 Model Architecture

### Feature Engineering
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 2000 features
- **N-grams**: Unigrams and bigrams (1, 2)
- **Stop words**: English (built-in)
- **Normalization**: L2 normalization

### Model
- **Algorithm**: Logistic Regression
- **Solver**: liblinear
- **Class weight**: Balanced (handles class imbalance)
- **Max iterations**: 200

### Data Split
- **Training set**: 80% (38,269 samples)
- **Test set**: 20% (9,568 samples)
- **Strategy**: Stratified split (maintains class distribution)

## 📊 Performance Metrics

### Overall Performance
| Metric | Score |
|--------|-------|
| Precision | 0.8502 |
| Recall | 0.8482 |
| F1-Score | 0.8486 |
| Accuracy | 84.82% |

### Per-Category Performance
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Purchase | 0.9289 | 0.9006 | 0.9145 | 493 |
| Internal Project | 0.7972 | 0.9269 | 0.8571 | 424 |
| Storage | 0.8741 | 0.9009 | 0.8873 | 555 |
| Access | 0.8922 | 0.8828 | 0.8875 | 1425 |
| HR Support | 0.8836 | 0.8346 | 0.8584 | 2183 |
| Hardware | 0.8372 | 0.8157 | 0.8263 | 2724 |
| Miscellaneous | 0.7926 | 0.8442 | 0.8176 | 1412 |
| Administrative rights | 0.7215 | 0.8097 | 0.7631 | 352 |

## 📁 Project Structure

```
ticket-classification/
│
├── Ticket_Classification.ipynb    # Main Jupyter notebook
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
│
├── final_classifier.pkl            # Trained model (generated)
├── final_results.json             # Performance metrics (generated)
├── detailed_analysis_report.txt   # Analysis report (generated)
│
└── visualizations/                 # Generated visualizations
    ├── visualization_performance_analysis.png
    └── visualization_priority_confidence.png
```

## 🔍 Dataset

- **Source**: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) (Kaggle)
- **Total Records**: 47,837 tickets
- **Features**: Text-based support tickets
- **Labels**: 8 categories
- **Quality Score**: 75%

### Data Characteristics
- No missing values after cleaning
- No duplicate records
- Average ticket length: 43.6 words
- Clean, preprocessed text data

## 🎨 Visualizations

The system generates comprehensive visualizations including:

1. **Performance Analysis Dashboard**: 9-panel comprehensive view
   - Category distribution
   - F1-Score by category
   - Precision vs Recall scatter plot
   - Metrics comparison
   - Test set composition
   - Performance summary
   - Top/bottom performers
   - Train vs Test accuracy

2. **Priority & Confidence Analysis**
   - Priority level distribution
   - Confidence score distribution

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time prediction API
- [ ] Deep learning models (BERT, transformers)
- [ ] Automated ticket routing
- [ ] SLA prediction
- [ ] Sentiment analysis
- [ ] Entity extraction (names, dates, products)
- [ ] Integration with ticketing systems (Zendesk, Jira, ServiceNow)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Seana Mutinda**
- Organization: Future Interns
- Project: ML Task 2 (2026)

## 🙏 Acknowledgments

- Dataset provided by [adisongoh](https://www.kaggle.com/adisongoh) on Kaggle
- Built with scikit-learn, pandas, and matplotlib
- Inspired by real-world IT support challenges

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the author.


**Note**: This is a machine learning project for educational and practical purposes. Performance may vary with different datasets and use cases. Always validate model predictions in production environments.
