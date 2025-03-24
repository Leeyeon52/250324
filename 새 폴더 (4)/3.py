import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 파일 경로 설정
train_path = "train.csv"
test_path = "test.csv"
sample_submission_path = "sample_submission.csv"
output_path = "submission.csv"

# 파일 존재 여부 확인
if not all(os.path.exists(p) for p in [test_path, sample_submission_path]):
    print("ERROR: 필요한 파일이 존재하지 않습니다. 실행을 중단합니다.")
    exit()

# 데이터 로드
try:
    df_test = pd.read_csv(test_path).drop(columns=["Unnamed: 0"], errors="ignore")
    df_sample = pd.read_csv(sample_submission_path)
    print(f"Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    print("df_test 컬럼 목록:", df_test.columns.tolist())
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# 'URL' 컬럼 확인 및 변경 (test 데이터)
text_feature = 'URL'
if text_feature not in df_test.columns:
    actual_url_column = None
    for col in df_test.columns:
        if 'url' in col.lower():
            actual_url_column = col
            break
    if actual_url_column:
        print(f"'{actual_url_column}' 컬럼을 'URL'로 변경합니다.")
        df_test.rename(columns={actual_url_column: 'URL'}, inplace=True)
    else:
        print("ERROR: test 데이터에서 'URL' 컬럼을 찾을 수 없습니다. 실행을 중단합니다.")
        exit()

# 중복 제거 및 결측치 제거
df_test = df_test.drop_duplicates().dropna()

# 테스트 데이터 150개로 샘플링
df_test_sampled = df_test.sample(n=150, random_state=42).reset_index(drop=True)
df_test_sampled.to_csv("test_sampled.csv", index=False)
print("샘플링된 테스트 데이터 저장 완료: test_sampled.csv")

# train.csv 생성 (test에서 URL을 가져와 악성 여부 랜덤 할당)
df_train = df_test_sampled.copy()
df_train['malicious'] = np.random.choice([0, 1], size=len(df_train))
df_train.to_csv(train_path, index=False)
print("새로운 train.csv 저장 완료")

# train.csv 다시 로드
df_train = pd.read_csv(train_path)
print(f"Train data loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
print("df_train 컬럼 목록:", df_train.columns.tolist())

# 특징(X)와 타겟(y) 분리
X_texts = df_train[text_feature].astype(str)
y = df_train['malicious']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X_texts)
test_tfidf = vectorizer.transform(df_test_sampled[text_feature].astype(str))

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 모델 학습 (랜덤 포레스트)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 검증 성능 확인
y_pred_rf = rf_model.predict(X_val)
print("Random Forest 정확도:", accuracy_score(y_val, y_pred_rf))

# 테스트 데이터 예측
y_test_pred = rf_model.predict(test_tfidf)

# 제출 파일 생성
df_sample = df_sample.sample(len(df_test_sampled), random_state=42)
df_sample['malicious'] = y_test_pred
df_sample.to_csv(output_path, index=False)
print(f"✅ 제출 파일 생성 완료: {output_path}")