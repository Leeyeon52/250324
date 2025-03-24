import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 파일 경로 설정
train_path = "train.csv"
test_sampled_path = "test_sampled.csv"
sample_submission_path = "sample_submission.csv"
output_path = "submission.csv"

# 파일 존재 여부 확인
if not all(os.path.exists(p) for p in [test_sampled_path, sample_submission_path]):
    print("ERROR: 필요한 파일이 존재하지 않습니다. 실행을 중단합니다.")
    exit()

# 데이터 로드
try:
    df_train = pd.read_csv(train_path).drop(columns=["Unnamed: 0"], errors="ignore")
    df_test = pd.read_csv(test_sampled_path).drop(columns=["Unnamed: 0"], errors="ignore")
    df_sample = pd.read_csv(sample_submission_path)
    print(f"Train data loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    print("Test Data Columns:", df_test.columns)
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# ID 컬럼 제거
df_train = df_train.drop(columns=["ID"], errors="ignore")
df_test = df_test.drop(columns=["ID"], errors="ignore")

# URL 컬럼 확인 및 설정
text_feature = 'URL'
if text_feature not in df_train.columns or text_feature not in df_test.columns:
    print(f"ERROR: '{text_feature}' 컬럼이 없습니다. 실행을 중단합니다.")
    print("현재 df_test 컬럼:", df_test.columns)
    exit()

# 악성 URL 분류 함수
def classify_url(url):
    malicious_keywords = ['.exe', '.zip', '.php', '@', '?', '=']
    return int(any(keyword in url for keyword in malicious_keywords))

df_train['malicious'] = df_train[text_feature].astype(str).apply(classify_url)

# 데이터 증강 (Augmentation)
def augment_url(url):
    if "http" in url:
        url = url.replace("http", "hxxp")  # 일부 변형 적용
    if "www" in url:
        url = url.replace("www", "wwx")
    if "?" in url:
        url += f"&random={random.randint(100, 999)}"  # 랜덤 쿼리 추가
    return url

df_train["augmented_url"] = df_train[text_feature].apply(augment_url)
df_train_augmented = df_train.copy()
df_train_augmented[text_feature] = df_train_augmented["augmented_url"]
df_train = pd.concat([df_train, df_train_augmented]).reset_index(drop=True)

# 특징(X)와 타겟(y) 분리
X_texts = df_train[text_feature].astype(str)
y = df_train['malicious']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X_texts)
test_tfidf = vectorizer.transform(df_test[text_feature].astype(str))

# SMOTE 적용 전 클래스 분포 확인
print("클래스 분포:", y.value_counts())
if y.value_counts().min() > 5:
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_tfidf, y = smote.fit_resample(X_tfidf, y)
    print("SMOTE 적용 완료. 새 클래스 분포:", y.value_counts())
else:
    print("⚠️ SMOTE 적용 불가: 한 클래스의 샘플 수가 너무 적음")

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 모델 학습 (랜덤 포레스트 & XGBoost)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(eval_metric="logloss")

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 검증 성능 확인
y_pred_rf = rf_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)
print("Random Forest 정확도:", accuracy_score(y_val, y_pred_rf))
print("XGBoost 정확도:", accuracy_score(y_val, y_pred_xgb))

# 테스트 데이터 예측
y_test_pred = rf_model.predict(test_tfidf)

# 제출 파일 생성
df_sample = df_sample.sample(len(df_test), random_state=42)
df_sample['malicious'] = y_test_pred
try:
    df_sample.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 생성 완료: {output_path}")
except Exception as e:
    print(f"Error saving submission file: {e}")
