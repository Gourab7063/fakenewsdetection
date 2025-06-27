# 📰 Fake News Detection using Machine Learning

This project focuses on detecting fake news using several supervised machine learning models. It was developed to address the growing issue of misinformation across digital platforms.

---

## 🧠 Objective

To classify news articles as **fake** or **real** based on their content using machine learning techniques. The project explores multiple classification models to determine the most effective approach.

---

## ⚙️ Technologies Used

- **Python**
- `pandas`, `numpy` – Data handling
- `scikit-learn` – ML models, TF-IDF vectorization, and evaluation
- `nltk` – Text preprocessing
- `matplotlib`, `seaborn` – Data visualization
- **Jupyter Notebook** – Interactive model development

---

## 🔍 ML Models Implemented

- ✅ Logistic Regression (LR)
- 🌲 Random Forest Classifier (RFC)
- 🌿 Gradient Boosting Classifier (GBC)
- 🌳 Decision Tree Classifier (DTC)

Each model was trained and evaluated to compare accuracy, precision, recall, and F1-score.

---

## 📌 Approach

1. **Data Collection:** Used a labeled dataset of real and fake news articles.
2. **Text Preprocessing:**
   - Lowercasing
   - Removing punctuation and stopwords
   - Tokenization and stemming
3. **Feature Extraction:** TF-IDF Vectorization
4. **Model Training:** Trained multiple classifiers on the feature vectors
5. **Model Evaluation:** Compared results using metrics like accuracy, confusion matrix, classification report

---

## 🏁 Sample Code Snippet

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

tfidf = TfidfVectorizer(max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)
predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## 📊 Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.92     |
| Random Forest      | 0.91     |
| Gradient Boosting  | 0.90     |
| Decision Tree      | 0.88     |

> ✅ Logistic Regression performed best in terms of overall accuracy and consistency.

---

## 🌐 Future Work

- Explore deep learning approaches like LSTM or BERT
- Create a web app using Streamlit or Flask
- Expand dataset with multilingual or cross-domain sources
- Add explainability using LIME or SHAP

---

## 🤝 Acknowledgements

Thanks to the open-source community and dataset providers for enabling this research.

---

## 👨‍💻 Author

- [Gourab7063](https://github.com/Gourab7063)

```
