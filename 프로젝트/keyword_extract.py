import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import os


# Matplotlib 한글 폰트 설정
from matplotlib import rc
import platform

if platform.system() == "Windows":
    rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # MacOS
    rc("font", family="AppleGothic")
else:  # Linux
    rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지

#  텍스트 전처리 함수
def clean_text(text):
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", "", text)  # 한글과 공백만 남기기
    text = re.sub(r"\s+", " ", text)  # 다중 공백 제거
    return text.strip()

# 불용어 제거 및 동사 제외 함수
def tokenize_and_remove_stopwords(text, stopwords):
    okt = Okt()
    tokens = okt.pos(text)  # 형태소 분석
    filtered_tokens = [word for word, pos in tokens if word not in stopwords and len(word) > 1 and pos != 'Verb']
    return " ".join(filtered_tokens)  # TF-IDF에 사용하기 위해 문자열로 반환

# 불용어 리스트 정의
stopwords = ["이", "은", "를", "에", "하다", "으로", "그리고", "사용", "기능", "는", "것", "있다", "등","이","가"]

# 시뮬레이션용 데이터


file_path = r'C:\Users\alswn\Desktop\새 폴더 (2)\데미안.txt'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

with open(file_path, 'r', encoding='utf-8') as file:
    documents = file.readlines() 

# 전처리 및 불용어 제거
processed_docs = []
for doc in documents:
    cleaned = clean_text(doc)  # 텍스트 정리
    tokenized = tokenize_and_remove_stopwords(cleaned, stopwords)  # 불용어 및 동사 제거
    processed_docs.append(tokenized)

# TF-IDF 계산
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)
keywords = vectorizer.get_feature_names_out()
scores = tfidf_matrix.toarray()

# 가장 중요한 키워드 추출 및 정리
# 문서 전체에서 평균 TF-IDF 점수가 높은 키워드 상위 10개 추출
keyword_scores = pd.DataFrame(scores, columns=keywords).mean().sort_values(ascending=False)
top_keywords = keyword_scores.head(10)

# 키워드와 점수를 CSV로 저장
top_keywords_df = top_keywords.reset_index()
top_keywords_df.columns = ['Keyword', 'TF-IDF Score']
top_keywords_df.to_csv("top_keywords.csv", index=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(top_keywords.index, top_keywords.values, color="skyblue")
plt.xlabel("TF-IDF Score")
plt.ylabel("Keywords")
plt.title("Top Keywords by TF-IDF Score")
plt.gca().invert_yaxis()  # 키워드 순서를 상위 점수가 위로 오도록 뒤집기
plt.show()
