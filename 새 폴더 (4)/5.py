import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 데이터 샘플링 (전체 데이터의 10% 사용)
X_train_sample = train_df["URL"].sample(frac=0.1, random_state=42)
y_train_sample = train_df["malicious"].loc[X_train_sample.index]

# 테스트 데이터 준비
X_test = test_df["URL"]

# 빈 데이터 프레임 생성 (결과 저장용)
submission_df = pd.DataFrame({'ID': test_df['ID']})

# TF-IDF + 로지스틱 회귀 모델
model = make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression())

# 모델 학습
model.fit(X_train_sample, y_train_sample)

# 예측 확률 계산
test_probabilities = model.predict_proba(X_test)
# Positive 클래스 확률을 가져옵니다 (일반적으로 열 인덱스 1이 positive)
test_predictions = test_probabilities[:, 1]

# 결과를 제출용 데이터프레임에 반영
submission_df['Label'] = test_predictions

# 예측 확률을 DataFrame에 추가
test_df['probability'] = test_predictions

# 저장할 파일 경로
submission_output_path = "./submission.csv"

# CSV 파일 저장
submission_df.to_csv(submission_output_path, index=False)

# 저장 완료 메시지 출력
print('Done.')
