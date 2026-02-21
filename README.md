🛍️ Advanced Product Similarity Engine (Flipkart Dataset)
An advanced content-based product recommendation system built using the Flipkart E-commerce Dataset, featuring:

🧰 Tech Stack
<p> <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg" /> <img src="https://img.shields.io/badge/Streamlit-Frontend-red.svg" /> <img src="https://img.shields.io/badge/Scikit--learn-ML-green.svg" /> <img src="https://img.shields.io/badge/Pandas-Data%20Processing-blue.svg" /> <img src="https://img.shields.io/badge/Matplotlib-Visualization-orange.svg" /> </p>

<img width="1915" height="985" alt="Screenshot 2026-02-22 031829" src="https://github.com/user-attachments/assets/772a60d4-bc40-40b8-8e2f-fe4b406cb423" />

<img width="1900" height="969" alt="Screenshot 2026-02-22 031925" src="https://github.com/user-attachments/assets/5e1e2f7a-bbf7-4790-bb8d-e80733893dbb" />

<img width="1899" height="985" alt="Screenshot 2026-02-22 031935" src="https://github.com/user-attachments/assets/cd9fa395-92c7-418f-8097-26413268442c" />

🧠 TF-IDF vector embeddings
🔁 Cosine similarity scoring
🏷️ Category-based filtering
🛍️ E-commerce style product card UI
📸 Product images
⭐ Price & ratings display
🔍 Smart search bar
📊 Cosine similarity bar graphs
🔥 Heatmap visualization for global similarity
🌐 Interactive Streamlit web interface
This project replicates the logic behind real e-commerce recommendation systems, such as those used by Flipkart and Amazon, built entirely using classical Machine Learning techniques.

🚀 Features
🔍 1. Smart Product Search
Search for products by typing any keyword - matches update instantly.

🛍️ 2. Product Cards UI
Each product is shown with a modern e-commerce style card:

Image
Brand
Price
Ratings
Category
Similarity score

🏷️ 3. Category Filtering
Recommendations stay within the same category, ensuring more accurate and meaningful suggestions.

🤖 4. TF-IDF + Cosine Similarity ML Engine
Converts text descriptions into vector representations
Computes pairwise similarity scores
Finds top-N most similar products
📊 5. Cosine Similarity Bar Graph
Displays similarity strength visually: █████████——— 0.81

🔥 6. Heatmap Visualization
A full heatmap of the similarity matrix showing:

Product clusters
High-correlation groups
Outliers
⭐ 7. Product Metadata Display
Each product includes:

Brand
Discounted/Retail Price
Overall Rating
Category
Image preview
🎨 8. Clean & Minimal Streamlit UI
A modern, intuitive interface for exploring recommendations.

🧠 Tech Stack
Machine Learning

Scikit-learn (TF-IDF, cosine similarity)
Pandas
NumPy
Frontend

Streamlit
Visualization

Matplotlib
Seaborn
Dataset

Flipkart E-commerce Dataset (cleaned)
🧹 Dataset Cleaning

The Flipkart raw dataset was processed to extract the following fields:

product_name

description

brand

prices

ratings

image URL

category tree

Image URLs stored in list format like:

["https://rukminim1.flixcart.com/..."]

were cleaned to:

https://rukminim1.flixcart.com/...

🔍 How the Recommendation Works

Step 1 — Preprocessing

Descriptions → lowercase → stopwords removed

Step 2 — TF-IDF Vectorization

Text converted into numerical vectors reflecting word importance.

Step 3 — Cosine Similarity

Measures similarity between product vectors.

Step 4 — Category-Based Filtering

Keeps results relevant and meaningful.

Step 5 — Visual Insights

Heatmap + bar graphs help understand similarity relationships.

✨ Future Enhancements

Image-based similarity (CNN embeddings)

Hybrid recommender system (text + image + metadata)

User behavior-based personalization

Product clustering dashboard

👤 Author
📧 Email: riddhisharma240604@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/riddhi-sharma10
💻 GitHub: https://github.com/riddhi-sharma10

If you have suggestions, improvements, or feedback - feel free to reach out or open an issue on GitHub.
I'm always open to learning, collaboration, and building better ML + Web projects!
