# 🧠 Quora Duplicate Question Pair Detection

This project addresses the problem of **identifying whether two questions asked on Quora are semantically similar or duplicates** of each other. It's based on feature engineering, vectorization, and supervised learning models such as **Random Forest** and **XGBoost**.

---

### 📉 Live Demo

Try the app here: [https://qpair-finder.streamlit.app/](https://qpair-finder.streamlit.app/)

---

### 📌 Project Overview

Duplicate questions create redundancy and clutter on platforms like Quora. This project helps in automatically detecting such duplicate question pairs using a combination of:

- Extensive **text preprocessing**
- Rich **feature engineering**
- **Bag-of-Words** vectorization
- Dimensionality reduction using **t-SNE**
- Classification using **Random Forest** and **XGBoost**

---

### 📂 Dataset

- Dataset used: `train.csv` from the [Quora Question Pairs Kaggle competition](https://www.kaggle.com/c/quora-question-pairs).
- Sample size: **30,000** random samples were taken for efficiency.

---

### ⚙️ Preprocessing Steps

- Lowercasing, punctuation removal, and special character replacements.
- HTML tag removal using `BeautifulSoup`.
- Expansion of contractions (e.g., `can't` → `cannot`).
- Feature generation:
  - Length-based features
  - Common word counts and ratios
  - Stopword-based token overlaps
  - Fuzzy similarity metrics using `fuzzywuzzy`
  - Longest common substring ratio using `distance.lcsubstrings`

---

### 📊 Feature Engineering

Key engineered features include:

| Feature Type     | Description |
|------------------|-------------|
| Basic            | Lengths, word counts, word share ratio |
| Token-based      | Common words/stops/tokens (min, max ratios), first/last word equality |
| Length-based     | Absolute difference, mean length, longest common substring ratio |
| Fuzzy Features   | Fuzzy ratio, partial ratio, token sort/set ratio |
| BOW Features     | CountVectorizer with max 3000 features on each question |

---

### 📈 Visualization

- Used `seaborn.pairplot()` to visualize feature distribution by target class (`is_duplicate`)
- Applied **t-SNE** (2D and 3D) to embed high-dimensional features into low-dimensional space
- Plotted with `matplotlib` and `plotly` for better interpretability

---

### 🤖 Models Used

- **Random Forest**
- **XGBoost Classifier**

Performance was evaluated using `accuracy_score` and `confusion_matrix`.

---

### 🧪 Inference Pipeline

- A function `query_point_creator(q1, q2)` allows prediction on new question pairs
- Trained models (`model.pkl`) and vectorizer (`cv.pkl`) are saved using `pickle`
- Custom test functions replicate preprocessing + feature generation steps for real-time inputs

---

### 🚀 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Abhavya-Singh02/Duplicate-Quora-Question_Pair.git
   cd Duplicate-Quora-Question_Pair
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open `Quora_Duplicate_Question_Analysis.ipynb` in Jupyter or Colab.

4. **Use trained model**:
   Load `model.pkl` and `cv.pkl` for inference.

---

### 📦 Requirements

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
bs4
nltk
distance
fuzzywuzzy
plotly
```

---

### 🧾 Sample Output

```python
q1 = "What is the capital of India?"
q2 = "Which city serves as the capital of India?"
query = query_point_creator(q1, q2)
model.predict(query)  # Output: [1] means duplicate
```

---

### 💡 Future Enhancements

- Implement Deep Learning with Siamese Networks or BERT
- Use word embeddings (Word2Vec, GloVe)
- Improve model generalization with cross-validation
- Deploy via Flask or FastAPI for web inference

---

### 📚 References

- [Quora Duplicate Question Kaggle Challenge](https://www.kaggle.com/c/quora-question-pairs)
- [Text Similarity Techniques](https://towardsdatascience.com/fuzzy-string-matching-4adaf0dfc48b)
