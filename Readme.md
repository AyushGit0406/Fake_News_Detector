#  Fake News Detector using LSTM

A deep learning-based web application that detects whether a news article is **FAKE** or **REAL** using an LSTM (Long Short-Term Memory) neural network.

---

##  Features

* LSTM-based text classification model
* Supports **headline + article body input**
* Displays:

  * Prediction (FAKE / REAL)
  * Confidence score
  * Probability breakdown
* Handles short/insufficient input intelligently
* Interactive UI built with Streamlit

---

## Project Structure

```
├── app.py                 # Streamlit web application
├── model.h5              # Trained LSTM model
├── tokenizer.pkl         # Tokenizer used during training
├── params.pkl            # Stores MAX_LEN parameter
├── requirements.txt      # Dependencies
├── Fake_News_Detector.ipynb  # Model training notebook
```

---

## How It Works

1. Input text (headline + optional body) is taken from user
2. Text is cleaned (removes URLs, punctuation, numbers, etc.)
3. Converted into sequences using tokenizer
4. Padded to fixed length
5. Passed into LSTM model
6. Output probability determines:

   * **REAL** (if > 0.5)
   * **FAKE** (if ≤ 0.5)

---

## Model Details

* Model: LSTM Neural Network
* Embedding Layer + Spatial Dropout
* LSTM (64 units)
* Dense layers with dropout
* Output: Sigmoid (binary classification)

---

## Performance

* Accuracy: **82.32%**
* Precision: **0.83**
* Recall: **0.82**
* F1 Score: **0.82**

---

## Installation

```bash
git clone https://github.com/AyushGit0406/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

---

## Requirements

* Python 3.11
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## Usage

* Enter a **headline (required)**
* Optionally add **article body**
* Click **Detect**
* Get prediction with confidence score

---

## Notes

* Very short input may not give reliable predictions
* Model performs better with full article text
* Confidence below 65% indicates uncertainty

---

##  Author

**Ayush Yadav**
M.Sc. Computer Science
NIT Tiruchirappalli


