# 📰 News Topic Classification System (AG News Benchmark)

A modular, high-performance Natural Language Processing (NLP) system that automatically categorizes news articles into four topical domains based on their **Title** and **Description**:
- 🌍 **Class 1: World**
- ⚽ **Class 2: Sports**
- 💼 **Class 3: Business**
- 🔬 **Class 4: Sci/Tech**

---

## 📊 Key Results Summary

Evaluated on the full provided test dataset (**7,600 news articles**, 1,900 balanced samples per class):

| Metric | Score | Details |
| :--- | :--- | :--- |
| **Test Accuracy** | **91.86%** | Overall top-1 classification accuracy |
| **Macro F1-Score** | **91.84%** | Unweighted average across all 4 categories |
| **Weighted F1-Score** | **91.84%** | Weighted by category support |
| **Inference Latency** | **< 1.0 ms / article** | ~1,100,000+ articles/sec throughput |

### Per-Category Performance Breakdown:

| Class ID | Category | Precision | Recall | F1-Score | Test Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **World** | **93.30%** | 90.11% | **91.67%** | 1,900 |
| **2** | **Sports** | **95.69%** | **98.26%** | **96.96%** | 1,900 |
| **3** | **Business** | **89.03%** | 88.42% | **88.72%** | 1,900 |
| **4** | **Sci/Tech** | **89.36%** | 90.63% | **89.99%** | 1,900 |

---

## 🏛️ Project Architecture

The codebase is organized into an industry-standard, modular ML structure:

```
gdg-task3/
├── app.py                             # Production FastAPI REST API & Web UI
├── train.py                           # Clean training & benchmarking CLI entry point
├── vercel.json                        # Vercel serverless deployment & route rewrites
├── .vercelignore                      # Vercel deployment exclusions
├── requirements.txt                   # Minimal project dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Standard version control ignore rules
│
├── api/                               # Serverless deployment entrypoints
│   └── index.py                       # Vercel ASGI serverless handler
│
├── data/                              # Datasets & data documentation
│   ├── train.csv                      # AG News training dataset (120,000 samples)
│   └── test.csv                       # AG News test dataset (7,600 samples)
│
├── models/                            # Serialized production models
│   └── news_classifier_pipeline.pkl   # Serialized Calibrated LinearSVC pipeline (pickle)
│
├── notebooks/                         # Exploratory Data Analysis & experiments
│   └── news_classification_pipeline.ipynb # Beginner-friendly tutorial & interactive notebook
│
├── public/                            # Web UI frontend
│   ├── index.html                     # Minimal, modern dark-mode interface
│   ├── style.css                      # Custom design tokens & glassmorphism styling
│   └── app.js                         # Dynamic frontend controller & API client
│
└── src/                               # Modular NLP package
    ├── __init__.py                    # Package exports (NewsClassifier, predict_article, run_pipeline)
    ├── config.py                      # Category labels, file paths, regex, hyperparameters
    ├── preprocessing.py               # Basic & advanced text cleaning, contractions, lemmatization
    ├── features.py                    # Word/Char TF-IDF vectorization and text metadata extractors
    ├── models.py                      # Candidate classifiers, pipelines, and Platt calibration wrappers
    ├── evaluate.py                    # Metrics calculation, confusion matrix, error analysis
    ├── predict.py                     # Production NewsClassifier inference engine
    ├── pipeline.py                    # Full training, benchmarking, and evaluation orchestrator
    └── schemas.py                     # Pydantic request & response validation schemas
```

---

## 🛠️ Methodology & Technical Details

### 1. Text Preprocessing Pipeline
The raw dataset contains several real-world formatting artifacts that degrade standard NLP models if not cleaned:
- **HTML Entity Repair:** Decodes `&amp;`, `&lt;`, `&gt;`, and repairs widespread malformed entities such as `#39;` (unescaped apostrophes) and backslash artifacts (`early\and` -> `early and`).
- **News Agency Boilerplate Removal:** Strips wire tags like `Reuters - `, `(AP)`, `AFP - `, `(Space.com)` to prevent models from learning spurious associations between news agencies and topics.
- **Contraction Expansion:** Normalizes informal and conversational English (`won't` -> `will not`, `it's` -> `it is`).
- **Title Weighting:** News headlines carry very dense semantic signals. We combine `Title` and `Description` with double emphasis on `Title` (`f"{Title} {Title}. {Description}"`).
- **Tokenization & Stopwords:** Custom domain stopwords combined with NLTK English stopwords, filtering repetitive wire jargon.
- **Lemmatization:** POS-aware WordNet lemmatizer with in-memory memoization cache for high throughput.

### 2. Feature Extraction & Engineering
- **Word & Bigram TF-IDF:** Configured with `ngram_range=(1, 2)` to capture both isolated terms and compound phrases (e.g., `wall street`, `prime minister`, `space telescope`).
- **Sublinear Term Frequency Scaling:** Replaces raw term frequency $tf$ with $1 + \log(tf)$ to prevent repetitive words from dominating topical signals.
- **Vocabulary Truncation:** Retains top 35,000 highly informative n-grams, discarding uninformative rare hapax legomena (`min_df=2`).

### 3. Model Exploration & Benchmarking

| Candidate Model | Test Accuracy | Macro F1 | Weighted F1 | Training Time | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 91.76% | 91.75% | 91.75% | 28.17s | ~1.3M docs/s |
| **LinearSVC (Champion)** | **91.97%** | **91.96%** | **91.96%** | **7.81s** | **~1.1M docs/s** |

### 4. Champion Model & Platt Probability Calibration
While **LinearSVC** achieved the highest test accuracy and fastest training time among discriminative models, raw SVMs only output uncalibrated hyperplane distances.

To provide well-calibrated class posterior probabilities $P(y = k \mid x)$ for real-world confidence scoring, the final champion model wraps `LinearSVC` with **Platt Scaling** via `CalibratedClassifierCV(cv=3, method='sigmoid')`.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the Jupyter Notebook
Open [`notebooks/news_classification_pipeline.ipynb`](file:///c:/Users/HP/OneDrive/Desktop/gdg-task3/notebooks/news_classification_pipeline.ipynb) in VS Code or JupyterLab.
It contains the step-by-step tutorial, data visualizations, model comparisons, and interactive prediction cells.

### 3. Train and Benchmark Models via CLI
```bash
# Standard full training on 120,000 articles
python train.py

# Fast iteration on a sample
python train.py --sample_size 10000

# Train with advanced lemmatization & stopword removal
python train.py --advanced_clean
```

### 4. Launch the Production FastAPI REST Service
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
### 5. Deploy to Vercel
```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod
```
The application will automatically serve the modern dark-mode Web UI, static assets, and the REST API at `/api/predict` and `/predict`.

### 7. Programmatic Inference in Python
```python
from src.predict import NewsClassifier

classifier = NewsClassifier()
result = classifier.predict(
    title="NASA James Webb Space Telescope Discovers Ancient Galaxies",
    description="Astronomers analyzed deep infrared spectra from the early universe."
)

print(result["category_name"])   # Output: Sci/Tech
print(result["confidence"])      # Output: 0.959 (95.9%)
print(result["probabilities"])   # Output: {'World': 0.015, 'Sports': 0.003, 'Business': 0.023, 'Sci/Tech': 0.959}
```

---

## 🔍 In-Depth Diagnostic Insights

### Top Predictive Keywords:
- **World:** `iraq`, `arafat`, `palestinian`, `minister`, `athens greece`, `kill`, `security council`
- **Sports:** `coach`, `cup`, `nascar`, `stadium`, `olympic`, `season`, `quarterback`, `victory`
- **Business:** `oil`, `investing`, `enron`, `halliburton`, `shares`, `revenue`, `profit`, `wall street`
- **Sci/Tech:** `nasa`, `internet`, `space`, `linux`, `software`, `microsoft`, `telescope`, `chip`

### Error Analysis & Boundary Overlaps:
The confusion matrix reveals that the vast majority of classification errors occur along intuitive semantic boundaries:
1. **Business vs. Sci/Tech (153 misclassifications):** Technology companies announcing corporate earnings, product delays, or antitrust lawsuits (e.g., *Intel postpones chip launch*, *Google IPO shares allocation*). Both categories share vocabulary like `company`, `shares`, `billion`, and `market`.
2. **World vs. Business (83 misclassifications):** Geopolitical events heavily involving trade, oil exports, or economic sanctions (e.g., *Venezuelan referendum and oil exports impact*).
