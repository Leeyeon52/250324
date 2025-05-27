import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


base_path = Path(__file__).resolve().parent / "open"
train_file = base_path / "train.csv"
test_file = base_path / "test.csv"
submission_file = base_path / "sample_submission.csv"


if not train_file.exists() or not test_file.exists() or not submission_file.exists():
    print(f" 필요한 데이터 파일이 존재하지 않습니다. 확인된 경로: {base_path}")
    exit()


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

print(f"Train Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")
print(f"Sample Submission Shape: {submission.shape}")


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_data.iloc[:, 1].astype(str).values)  
X = train_data.iloc[:, 2:].values.astype(np.float32)  
test_X = test_data.iloc[:, 1:].values.astype(np.float32)  


X /= 255.0
test_X /= 255.0


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(train_dataset, epochs=10, validation_data=val_dataset)


y_pred = model.predict(test_X)
y_pred_labels = np.argmax(y_pred, axis=1)


y_pred_labels = label_encoder.inverse_transform(y_pred_labels)


submission['label'] = y_pred_labels
submission_path = base_path / "baseline_submission.csv"
submission.to_csv(submission_path, index=False, encoding='utf-8-sig')

print(f" 제출 파일 저장 완료: {submission_path}")
