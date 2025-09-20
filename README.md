# 🎬 Comparative Sentiment Analysis on Movie Reviews  

This project applies **machine learning algorithms** to analyze IMDB movie reviews and classify them as **positive, negative, or neutral**.  
The focus is on comparing algorithms in terms of accuracy, speed, and effectiveness for text classification.  

---

## 📊 Overview
- **Dataset:** [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Goal:** Analyze customer reviews to understand sentiments and market behavior for better customer experience.  
- **Algorithms Implemented:**  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Multinomial Naive Bayes  
  - K-Nearest Neighbors (KNN)  

---

## 🛠️ Data Preprocessing
To prepare raw text for machine learning:  
1. **Cleaning HTML & special characters** – remove tags, brackets, and symbols.  
2. **Stemming** – reduce words to their root form (e.g., *cool, cooler, coolest → cool*).  
3. **Stop word removal** – remove frequent but uninformative words (*a, an, the, is, etc.*).  
4. **Splitting** – divide dataset into training and testing sets.  

---

## ⚙️ Algorithms & Workflow

### 1. Logistic Regression
- Supervised classification algorithm predicting binary outcomes.  
- Steps: Split data → Fit Logistic Regression → Predict → Evaluate.  
- Imported from `scikit-learn`.  

### 2. Support Vector Machine (SVM)
- Finds a hyperplane to separate classes.  
- Uses `sklearn.svm` for implementation.  
- Flexible with kernels and tuning options.  

### 3. Multinomial Naive Bayes
- Probabilistic algorithm widely used in NLP.  
- Based on Bayes theorem.  
- Steps: Preprocess → Fit NB → Predict → Evaluate (Confusion Matrix).  

### 4. K-Nearest Neighbors (KNN)
- Classifies based on majority vote among *k* nearest neighbors.  
- Works well for small datasets but computationally expensive for large text features.  
- Steps: Split → Build k-NN model → Train → Predict.  

---

## 📈 Results

| Classifier              | Accuracy | Execution Time |
|--------------------------|----------|----------------|
| Logistic Regression      | *89%*| 64 sec         |
| Support Vector Machine   | **89%** | 3 sec          |
| Multinomial Naive Bayes  | 87%      | 1 sec          |
| K-Nearest Neighbors      | 77%      | 76 sec         |

🔎 **Insights:**  
- Logistic Regression and SVM achieved the best accuracy (89%).  
- SVM was the fastest (3 sec) compared to Logistic Regression (64 sec).  
- Naive Bayes performed slightly lower (87%) but with extremely fast execution.  
- KNN lagged in both accuracy and speed for this high-dimensional dataset.  

---

## 📂 Repository Structure

Comparative-Sentimental-Analysis-On-Movie-Reviews/
├── movie_dat_imdb.py      # Main script for preprocessing, training, and evaluation
├── README.md              # Project documentation


---

## 🔮 Future Work
- Extend to **deep learning models** (e.g., LSTMs, Transformers, BERT).  
- Add **neutral sentiment** for finer classification.  
- Perform hyperparameter tuning (grid search, random search).  
- Modularize into multiple scripts (`data_preprocessing.py`, `train.py`, `evaluate.py`).  
- Deploy as a Flask/FastAPI service for real-time predictions.  

---

## 📜 License
MIT © 2025 Teenu Anand Nukavarapu  
