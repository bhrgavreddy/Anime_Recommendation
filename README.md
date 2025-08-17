# Anime Recommendation System 🎌✨

This project implements an **Anime Recommendation System** using the following techniques:

- 🎭 **Content-Based Filtering**  
- 👥 **Collaborative Filtering**  
- 🧬 **Hybrid Recommendation** (combining both)

The system recommends anime titles either based on similarity of content or user preferences, or a combination of both.

---

## 📁 Dataset Files

The following CSV files are used as input:

- **`anime.csv`**  
  Contains basic information about anime:
  - `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members`

- **`rating_complete.csv`**  
  Contains user ratings for anime:
  - `user_id`, `anime_id`, `rating`  
  > ⚠️ *This is a reduced version for manageability. The original dataset contains over 57 million ratings.*

- **`anime_with_synopsis.csv`**  
  Extended anime data including:
  - `anime_id`, `name`, `genre`, `synopsis`, etc.

---

## 🎯 Recommendation Outputs

### 📚 1. Content-Based Filtering
- Recommends anime with similar **synopses** to a given anime.
- **Input:** Anime name (e.g., `"Naruto"`)
- **Output:** Top similar anime titles based on text similarity.

### 🧑‍🤝‍🧑 2. Collaborative Filtering
- Recommends anime based on a **user’s historical ratings** using matrix factorization.
- **Input:** `user_id` (e.g., `3`)
- **Output:** Personalized anime suggestions.

### ⚖️ 3. Hybrid Recommendation
- Combines **content similarity** and **user preferences**.
- Balances both techniques to output a refined list of recommendations.
- Suitable when partial user data is available.
