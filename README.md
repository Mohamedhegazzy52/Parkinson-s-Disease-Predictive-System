[README.md](https://github.com/user-attachments/files/22034853/README.md)
Parkinson’s Disease Prediction using Machine Learning

Overview
This project implements a **Parkinson’s Disease detection system** using **machine learning (SVM)**.  
The dataset contains biomedical voice measurements, and the goal is to predict whether a person has Parkinson’s Disease (`status = 1`) or not (`status = 0`).

The system:
- Loads and preprocesses the dataset.  
- Trains a **Support Vector Machine (SVM)** model.  
- Evaluates accuracy on training and testing sets.  
- Provides a **predictive system** to check new patient samples.  


Dataset
- The dataset used is parkinsons.csv  
- Features are voice measurements such as fundamental frequency, jitter, shimmer, noise-to-harmonics ratio, and nonlinear measures.  

| Column | Description |
|--------|-------------|
| name | Unique identifier for each patient sample |
| MDVP:Fo(Hz) | Average vocal fundamental frequency |
| MDVP:Fhi(Hz) | Maximum vocal fundamental frequency |
| MDVP:Flo(Hz) | Minimum vocal fundamental frequency |
| MDVP:Jitter(%) | Variation in fundamental frequency |
| MDVP:Jitter(Abs) | Absolute jitter |
| MDVP:RAP | Relative average perturbation |
| MDVP:PPQ | Pitch period perturbation quotient |
| Jitter:DDP | Variation measure of pitch |
| MDVP:Shimmer | Variation in amplitude |
| MDVP:Shimmer(dB) | Amplitude variation in dB |
| Shimmer:APQ3 | Amplitude perturbation quotient (3 cycles) |
| Shimmer:APQ5 | Amplitude perturbation quotient (5 cycles) |
| MDVP:APQ | General amplitude perturbation quotient |
| Shimmer:DDA | Difference of amplitude measures |
| NHR | Noise-to-harmonics ratio |
| HNR | Harmonics-to-noise ratio |
| status | Target variable → 0 = Healthy, 1 = Parkinson’s |
| RPDE | Recurrence period density entropy |
| DFA | Detrended fluctuation analysis |
| spread1 | Nonlinear dynamic spread measure |
| spread2 | Alternative nonlinear spread measure |
| D2 | Correlation dimension |
| PPE | Pitch period entropy |



Project Workflow

1. Install Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Load Data
```python
parkinson_data = pd.read_csv('/content/parkinsons.csv')
print(parkinson_data.head())
```

### 3. Preprocess Data
- Drop `name` column.  
- Split into features (X) and labels (Y).  
- Standardize data with `StandardScaler`.  

### 4. Train Model
```python
Model = svm.SVC(kernel='linear')
Model.fit(X_train, Y_train)
```

### 5. Evaluate Model
```python
X_train_pred = Model.predict(X_train)
print("Training Accuracy:", accuracy_score(Y_train, X_train_pred))

X_test_pred = Model.predict(X_test)
print("Testing Accuracy:", accuracy_score(Y_test, X_test_pred))
```

### 6. Predict New Data
```python
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,
              0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,
              26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

input_data_as_numpy = np.asarray(input_data).reshape(1,-1)
std_data = scaler.transform(input_data_as_numpy)

prediction = Model.predict(std_data)
print("Parkinson’s" if prediction[0] == 1 else "Healthy")
```

---

## 📊 Example Results
- **Training Accuracy:** ~0.85 – 0.95 (varies by run).  
- **Testing Accuracy:** ~0.80 – 0.90 (varies by run).  

---

## 🚀 Future Improvements
- Use **cross-validation** for more robust accuracy.  
- Try **different classifiers** (Random Forest, Gradient Boosting, Neural Networks).  
- Deploy model with **Flask/Streamlit** for real-time prediction.  

---

## 📜 License
This project is for **educational and research purposes only**.  
It does **not** use or expose real patient data.  
