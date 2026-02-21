import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['description'])

        # Entire similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_similar_products(self, index, top_n=5):
        category = self.data.iloc[index]["category"]

        # Filter products with same category
        same_cat_indices = self.data[self.data["category"] == category].index.tolist()

        scores = []
        for idx in same_cat_indices:
            scores.append((idx, self.similarity_matrix[index][idx]))

        # Sort by similarity
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Skip itself
        scores = [s for s in scores if s[0] != index]

        top_items = scores[:top_n]

        results = []
        for idx, score in top_items:
            row = self.data.iloc[idx]
            desc = str(row.get("description", "") or "").strip()
            results.append({
                "name":         row["product_name"],
                "image":        row["image"],
                "brand":        row["brand"],
                "price":        row["discounted_price"],
                "retail_price": row.get("retail_price", None),
                "rating":       row["overall_rating"],
                "score":        round(score, 3),
                "category":     row["category"],
                "description":  desc,
            })

        return results