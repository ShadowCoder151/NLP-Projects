from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import load
import numpy as np
import pandas as pd


train = load('train_preprocessed.joblib')
valid = load('valid_preprocessed.joblib')


def combine_features(row):
    entity = np.array(row['entity'], dtype=float).reshape(-1)
    content = np.array(row['content'], dtype=float).reshape(-1)
    return np.concatenate([entity, content])


X_train = np.array(train.apply(combine_features, axis=1).tolist())
y_train = np.array(train['sentiment'].tolist())

X_test = np.array(valid.apply(combine_features, axis=1).tolist())
y_test = np.array(valid['sentiment'].tolist())


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=5000,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)


y_predict = model.predict(X_test_scaled)
print(classification_report(y_test, y_predict))
