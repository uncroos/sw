import pandas as pd

# 데이터 로드
data = pd.read_csv('/Users/an-yohan/Documents/GitHub/SW/data/output/submission.csv')  # 파일명을 적절히 수정하세요.

# 'real'과 'fake'를 비교하여 'result' 열 생성
data['result'] = (data['real'] > data['fake']).astype(int)

# 결과 저장
data.to_csv('submission_with_result.csv', index=False)

# 처리된 데이터 확인
print(data.head())
