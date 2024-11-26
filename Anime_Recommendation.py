import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

anime_df = pd.read_csv('anime.csv')  
ratings_df = pd.read_csv('rating_complete.csv') 
synopsis_df = pd.read_csv('anime_with_synopsis.csv') 

anime_df = anime_df.dropna(subset=['Name'])  
anime_df = anime_df.set_index('anime_id') 
synopsis_df['synopsis'] = synopsis_df['synopsis'].fillna('') 

def get_content_based_recommendations(Name, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(synopsis_df['synopsis'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(synopsis_df.index, index=synopsis_df['Name']).drop_duplicates()
    idx = indices.get(Name)
    if idx is None:
        return f"'{Name}' not found in the dataset."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    anime_indices = [i[0] for i in sim_scores]
    return synopsis_df.iloc[anime_indices][['Name', 'synopsis']].reset_index(drop=True)

def get_collaborative_recommendations(user_id, top_n=10):
    anime_counts = ratings_df['anime_id'].value_counts()
    popular_anime = anime_counts[anime_counts >= 50].index
    filtered_ratings = ratings_df[ratings_df['anime_id'].isin(popular_anime)]
    user_counts = filtered_ratings['user_id'].value_counts()
    active_users = user_counts[user_counts >= 10].index
    filtered_ratings = filtered_ratings[filtered_ratings['user_id'].isin(active_users)]
    user_id_map = filtered_ratings['user_id'].astype('category')
    anime_id_map = filtered_ratings['anime_id'].astype('category')
    row = user_id_map.cat.codes
    col = anime_id_map.cat.codes
    data = filtered_ratings['rating']
    sparse_user_ratings = csr_matrix((data, (row, col)))
    if user_id not in user_id_map.cat.categories:
        return f"User ID {user_id} not found in the dataset."
    user_idx = list(user_id_map.cat.categories).index(user_id)
    user_sim = cosine_similarity(sparse_user_ratings, sparse_user_ratings[user_idx])
    similar_users = np.argsort(-user_sim.flatten())[1:top_n + 1]
    similar_users_ratings = sparse_user_ratings[similar_users, :]
    weighted_ratings = similar_users_ratings.T.dot(user_sim[similar_users].flatten())
    recommendation_scores = weighted_ratings / np.abs(user_sim[similar_users].flatten()).sum()
    recommendations = pd.Series(recommendation_scores, index=anime_id_map.cat.categories).sort_values(ascending=False)
    recommended_anime_ids = recommendations.head(top_n).index
    recommended_anime = anime_df.loc[recommended_anime_ids].reset_index(drop=True)
    return recommended_anime[['Name', 'genre']]

def get_hybrid_recommendations(Name, user_id, top_n=10, alpha=0.5):
    content_recommendations = get_content_based_recommendations(Name, top_n=top_n * 2)
    if isinstance(content_recommendations, str):
        return content_recommendations
    collaborative_recommendations = get_collaborative_recommendations(user_id, top_n=top_n * 2)
    if isinstance(collaborative_recommendations, str):
        return collaborative_recommendations
    content_recommendations['score'] = 1 - content_recommendations.index / len(content_recommendations)
    collaborative_recommendations['score'] = 1 - collaborative_recommendations.index / len(collaborative_recommendations)
    content_recommendations = content_recommendations.set_index('Name')
    collaborative_recommendations = collaborative_recommendations.set_index('Name')
    hybrid_scores = (
        alpha * content_recommendations['score'].reindex(anime_df['Name'], fill_value=0) +
        (1 - alpha) * collaborative_recommendations['score'].reindex(anime_df['Name'], fill_value=0))
    hybrid_recommendations = hybrid_scores.sort_values(ascending=False).head(top_n)
    final_recommendations = anime_df.loc[anime_df['Name'].isin(hybrid_recommendations.index)][['Name', 'genre']]
    return final_recommendations.reset_index(drop=True)

anime_title_example = input("Enter the anime name you like : ").strip()
anime_title_example = anime_title_example if anime_title_example else None

user_id_example_input = input("Enter your user ID : ").strip()
user_id_example = int(user_id_example_input) if user_id_example_input.isdigit() else None

print("\nContent-Based Recommendations:")
print(get_content_based_recommendations(anime_title_example))

print("\nCollaborative Recommendations:")
print(get_collaborative_recommendations(user_id_example))

print("\nHybrid Recommendations:")
print(get_hybrid_recommendations(anime_title_example, user_id_example))
