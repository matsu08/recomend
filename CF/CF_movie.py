#ライブラリを読み込む
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

#データを読み込む
with open('data/ratings.csv', 'r', encoding = 'utf-8') as f:
    df_rating = pd.read_csv(f)
with open('data/movies.csv', 'r', encoding = 'utf-8') as f:
    df_movie = pd.read_csv(f)
#timestampを削除
df_raring = df_rating.drop('timestamp', axis = 1)
#ピボットテーブルを作成し，それをスパース行列に変換する
df_pivot = df_rating.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
df_sparse = csr_matrix(df_pivot.values)
#コサイン類似度を計算
cos_sim = cosine_similarity(df_sparse)
cos_sim_df = pd.DataFrame(cos_sim, index = df_pivot.index, columns = df_pivot.index)
#ユーザーIDを入力して類似度の高い上位20名のユーザーをリストに格納
user_id = int(input('ユーザーIDを入力してください:'))
sim_users = cos_sim_df.iloc[user_id - 1,:].sort_values(ascending=False)[0:20]
sim_users_list = sim_users.index.tolist()
#類似度の高い上位ユーザーのスコア情報を集めてデータフレームに格納
sim_df = pd.DataFrame()
for user in sim_users_list:
    sim_df = pd.concat([sim_df, pd.DataFrame(df_pivot.loc[user]).T])
#対象ユーザーの視聴済み映画を取得
watched_movies = df_raring[df_raring['userId'] == user_id]['movieId'].tolist()
#未視聴の映画リストを作成
unseen_films = list(set(df_pivot.columns) - set(watched_movies))
#未視聴のアニメの平均評点を計算
result = []
for i in range(len(sim_df.columns)):
    mean_score = sim_df.iloc[:,i].mean()
    name = sim_df.columns[i]
    result.append([name, mean_score])
#集計結果のうちスコアの高い上位20番までを格納
result_df = pd.DataFrame(result, columns = ['movieId', 'score']).sort_values('score', ascending=False)[0:20]
#映画名を取得
result_df = pd.merge(result_df, df_movie, on='movieId', how='left')
#genre,movieIDを落として映画名のみを表示させる
result_df = result_df.drop(['genres', 'movieId'], axis=1)
#インデックスを１から振り直す
result_df.index = np.arange(1, len(result_df)+1)
#結果を表示
print(result_df)
