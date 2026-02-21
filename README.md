🛍️ Advanced Product Similarity Engine (Flipkart Dataset)

🧰 Tech Stack
<p> <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg" /> <img src="https://img.shields.io/badge/Streamlit-Frontend-red.svg" /> <img src="https://img.shields.io/badge/Scikit--learn-ML-green.svg" /> <img src="https://img.shields.io/badge/Pandas-Data%20Processing-blue.svg" /> <img src="https://img.shields.io/badge/Matplotlib-Visualization-orange.svg" /> </p>

A content-based product recommendation system built using classical Machine Learning techniques, replicating the logic behind real-world e-commerce recommenders like Flipkart and Amazon.
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

<img width="1915" height="985" alt="Screenshot 2026-02-22 031829" src="https://github.com/user-attachments/assets/772a60d4-bc40-40b8-8e2f-fe4b406cb423" />

<img width="1900" height="969" alt="Screenshot 2026-02-22 031925" src="https://github.com/user-attachments/assets/5e1e2f7a-bbf7-4790-bb8d-e80733893dbb" />

<img width="1899" height="985" alt="Screenshot 2026-02-22 031935" src="https://github.com/user-attachments/assets/cd9fa395-92c7-418f-8097-26413268442c" />

✨ Features
🔍 Smart Product Search
Type any keyword to instantly find and select products from the dataset.
🛍️ E-Commerce Style Product Cards
Each recommendation is displayed as a modern product card showing:

Product image
Brand name
Discounted & retail price
Overall rating
Category
Cosine similarity score

🏷️ Category-Based Filtering
Recommendations are filtered to the same category as the query product, ensuring relevant and meaningful results.
🤖 TF-IDF + Cosine Similarity Engine

Converts product descriptions into TF-IDF vector representations
Computes pairwise cosine similarity scores
Returns the top-N most similar products

📊 Cosine Similarity Bar Graph
Displays a ranked bar chart of similarity scores for the recommended products.
🔥 Global Similarity Heatmap
A full heatmap of the similarity matrix across all products, revealing:

Product clusters
High-correlation groups
Outliers


🧰 Tech Stack
LayerToolsMachine LearningScikit-learn (TF-IDF, Cosine Similarity), NumPy, PandasFrontendStreamlitVisualizationMatplotlib, SeabornDatasetFlipkart E-commerce Dataset (cleaned)

🗂️ Project Structure
flipkart-similarity-engine/
│
├── app.py                  # Main Streamlit application
├── recommender.py          # TF-IDF + cosine similarity logic
├── data/
│   └── flipkart_cleaned.csv  # Cleaned dataset
├── utils/
│   └── preprocessing.py    # Text cleaning & feature extraction
├── requirements.txt
└── README.md

🧹 Dataset Preprocessing
The raw Flipkart dataset was cleaned to extract the following fields:
FieldDescriptionproduct_nameName of the productdescriptionFull product descriptionbrandBrand namediscounted_priceSale priceretail_priceOriginal priceoverall_ratingAverage ratingcategoryProduct categoryimageProduct image URL
Image URLs stored as Python lists (e.g., ["https://rukminim1.flixcart.com/..."]) were cleaned to plain strings.

🧠 How It Works
Step 1 — Text Preprocessing
  Product descriptions → lowercased → stopwords removed → cleaned tokens

Step 2 — TF-IDF Vectorization
  Cleaned text → numerical vectors reflecting word importance across the corpus

Step 3 — Cosine Similarity
  Dot product of normalized vectors → similarity score between 0 and 1

Step 4 — Category Filtering
  Results filtered to the same category as the query product

Step 5 — Visual Output
  Top-N results displayed as cards + bar graph + global heatmap

🚀 Getting Started
Prerequisites

Python 3.8+
pip

Installation
bash# Clone the repository
git clone https://github.com/riddhi-sharma10/flipkart-similarity-engine.git
cd flipkart-similarity-engine

# Install dependencies
pip install -r requirements.txt
Run the App
bashstreamlit run app.py
The app will open in your browser at http://localhost:8501.

📦 Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn

Add these to your requirements.txt and install with pip install -r requirements.txt.

🔮 Future Enhancements

 Image-based similarity using CNN embeddings
 Hybrid recommender system combining text + image + metadata
 User behavior-based personalization
 Product clustering dashboard
 Deployment on Streamlit Cloud / HuggingFace Spaces

👤 Author
📧 Email: riddhisharma240604@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/riddhi-sharma10
💻 GitHub: https://github.com/riddhi-sharma10

If you have suggestions, improvements, or feedback - feel free to reach out or open an issue on GitHub.
I'm always open to learning, collaboration, and building better ML + Web projects!
