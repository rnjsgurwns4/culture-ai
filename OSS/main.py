from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from recommend_model import get_hybrid_recommendations
import subprocess
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

app = FastAPI(title="Hybrid Recommendation API")

# 요청 Body 형식 정의
class RecommendationRequest(BaseModel):
    user_id: int

# 응답 Body 형식 정의
class RecommendationResponse(BaseModel):
    recommended_contents: List[int]
    
class IntRequest(BaseModel):
    value: int

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    contents = get_hybrid_recommendations(request.user_id, top_n_each=3)
    
    if len(contents) == 3:
        for i in range(1, 4):
            contents.append(i)

    return {"recommended_contents": contents}

@app.post("/run-model")
def run_notebook(data: IntRequest):
    if data.value == 1:
        
        try:
            result1 = subprocess.run(
            ["python", "XGBoost_mysql연동.py"],
            capture_output=True,  # stdout/stderr 모두 잡기
            text=True,            # 출력 문자열로 받기
            check=True            # 오류 발생 시 예외 던짐
        )
            result2 = subprocess.run(
            ["python", "Surprise_mysql연동.py"],
            capture_output=True,  # stdout/stderr 모두 잡기
            text=True,            # 출력 문자열로 받기
            check=True            # 오류 발생 시 예외 던짐
        )
            
            return {"message": "성공", "xgboost_output": result1.stdout,
    "surprise_output": result2.stdout}

        except subprocess.CalledProcessError as e:
            
            return {"message": "실패", "xgboost_output": e.stdout}
    return 0;

# 새벽 1시에 모델_mysql연동.py 실행하는 함수
def scheduled_job():
    print("스케줄러 실행: 모델 실행")
    try:
        result1 = subprocess.run(
            ["python", "XGBoost_mysql연동.py"],
            capture_output=True,
            text=True,
            check=True
        )
        result2 = subprocess.run(
        ["python", "Surprise_mysql연동.py"],
        capture_output=True,  # stdout/stderr 모두 잡기
        text=True,            # 출력 문자열로 받기
        check=True            # 오류 발생 시 예외 던짐
        )
        
        print("스케줄러 정상 실행됨:")
        print(result1.stdout)
        print(result2.stdout)
    except subprocess.CalledProcessError as e:
        print("스케줄러 오류 발생:")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

# APScheduler 설정 및 실행
scheduler = AsyncIOScheduler()
# 매일 새벽 1시 (01:00)에 실행
scheduler.add_job(scheduled_job, 'cron', hour=20, minute=44)
scheduler.start()

# uvicorn을 asyncio 이벤트 루프에서 실행 시켰을 때 scheduler가 작동함을 보장하기 위해 빈 async 함수 실행
@app.on_event("startup")
async def startup_event():
    print("서버 시작 - 스케줄러 작동 대기 중")
    # 이벤트 루프에 잡 추가 등 필요 시 처리 가능