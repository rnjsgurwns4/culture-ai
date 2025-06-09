#!/usr/bin/env python
# coding: utf-8

# In[1]:


# db_config.py

import pymysql

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='kwon0822@',
        db='culture_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


# In[2]:


# main.py

import pandas as pd
from db_config import get_connection

# MySQL 연결
conn = get_connection()

# 각 테이블을 pandas DataFrame으로 불러오기
members_df = pd.read_sql("SELECT * FROM member", conn)
contents_df = pd.read_sql("SELECT * FROM content_detail", conn)
likes_df = pd.read_sql("SELECT * FROM content_favorite", conn)
region_df = pd.read_sql("SELECT * FROM region_coords", conn)
subcategory_df = pd.read_sql("SELECT * FROM content_sub_category", conn)
category_df = pd.read_sql("SELECT * FROM content_category", conn)

# 연결 종료
conn.close()


# In[3]:


# 데이터 확인
print("members_df:", members_df.shape)
print("contents_df:", contents_df.shape)
print("likes_df:", likes_df.shape)
print("region_df:", region_df.shape)
print("subcategory_df:", subcategory_df.shape)
print("category_df:", category_df.shape)


# In[4]:


# 필요한 컬럼 선택
user_features = members_df[['id', 'age', 'gender', 'location', 'keyword1', 'keyword2', 'keyword3']].copy()

# 범주형 인코딩
user_features_encoded = pd.get_dummies(user_features.set_index('id'))

# 인덱스: member_id / 값: 벡터
user_feature_matrix = user_features_encoded.sort_index()


# In[5]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# 유사도 행렬 계산 (index 순서 기준)
user_sim_matrix = cosine_similarity(user_feature_matrix)

# 유사도 행렬을 DataFrame으로 변환 (행/열: member_id)
user_ids = user_feature_matrix.index.tolist()
user_sim_df = pd.DataFrame(user_sim_matrix, index=user_ids, columns=user_ids)

print("사용자 유사도 행렬 (일부):\n", user_sim_df.iloc[:5, :5])


# In[19]:


TOP_K = 5   # 유사 사용자 수
TOP_N = 3   # 추천 콘텐츠 수

recommendations = []

# 모든 사용자에 대해 반복
for user_id in user_sim_df.index:
    # 본인을 제외한 유사 사용자 Top-K 추출
    similar_users = (
        user_sim_df.loc[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
        .head(TOP_K)
        .index.tolist()
    )
    
    # 유사 사용자가 찜한 콘텐츠들 집합
    sim_users_likes = likes_df[likes_df['member_id'].isin(similar_users)]
    liked_contents = sim_users_likes['content_detail_id'].value_counts()
    
    # 현재 사용자가 이미 찜한 콘텐츠
    user_liked = set(likes_df[likes_df['member_id'] == user_id]['content_detail_id'])
    
    # 내가 찜하지 않은 콘텐츠 중에서 높은 순으로 추천
    for content_id, count in liked_contents.items():
        if content_id not in user_liked:
            recommendations.append({
                'member_id': user_id,
                'content_detail_id': content_id,
                'like_count': count
            })
            if sum(r['member_id'] == user_id for r in recommendations) >= TOP_N:
                break  # 3개만 추천

# 결과 DataFrame
recommend_df = pd.DataFrame(recommendations)
print("사용자별 추천 결과 (샘플):\n", recommend_df.head(20))


# In[21]:


recommend_df.rename(columns={'like_count': 'score'}, inplace=True)
recommend_df.to_csv("collaborative_recommendations.csv", index=False)

