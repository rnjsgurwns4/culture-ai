# culture-ai
2025 공소프로젝트  

## 파이썬 필요 라이브러리 설치 명령어:

pip install fastapi pydantic uvicorn apscheduler

pip install pymysql pandas numpy scikit-learn gensim xgboost  


## fastapi 실행 명령어:

OSS 파일에 들어가서

uvicorn main:app --reload  


## OS에 따라서 오류 발생 가능성 있음

-> main.py 38, 44, 63, 69 번째 줄:

window: ["python", ""]

linux: ["python3", ""]


## 주의사항

db_config.py랑 XGBoost_mysql연동.py, Surprise_mysql연동.py 윗부분에 mysql 비밀번호 변경 필수
