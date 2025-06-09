import pandas as pd

# 두 모델 결과 파일 불러오기
xgb_df = pd.read_csv("xgboost_recommendations.csv")        # XGBoost 모델 결과
collaborative_df = pd.read_csv("collaborative_recommendations.csv")  # Surprise 모델 결과

# ✅ 하이브리드 추천 함수
def get_hybrid_recommendations(user_id: int, top_n_each: int = 3):
    # XGBoost 추천 추출
    xgb_top = (
        xgb_df[xgb_df['member_id'] == user_id]
        .sort_values(by='final_score', ascending=False)
        .head(top_n_each)['content_detail_id']
        .tolist()
    )

    # Surprise 추천 추출
    collaborative_top = (
        collaborative_df[collaborative_df['member_id'] == user_id]
        .sort_values(by='score', ascending=False)
        .head(top_n_each)['content_detail_id']
        .tolist()
    )

    return xgb_top + collaborative_top  # 앞 3개는 xgb, 뒤 3개는 surprise
