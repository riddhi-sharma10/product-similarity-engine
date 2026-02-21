🛍️ Product Similarity Engine – Flipkart Dataset
<p align="center"> <img src="assets/banner.png" width="80%" /> </p>

A high-performance content-based recommendation engine built using
TF-IDF · Cosine Similarity · Streamlit on the Flipkart E-commerce Dataset.

📸 Screenshots

<img width="1915" height="985" alt="Screenshot 2026-02-22 031829" src="https://github.com/user-attachments/assets/00a470de-9905-4f26-88f9-1b9f3bab1c02" />
<img width="1900" height="969" alt="Screenshot 2026-02-22 031925" src="https://github.com/user-attachments/assets/df3898be-8899-40ca-b36b-234b2e7a9bb4" />
<img width="1899" height="985" alt="Screenshot 2026-02-22 031935" src="https://github.com/user-attachments/assets/4a9a2d1c-c77b-406e-b76e-73c51d020f0e" />
<img width="1819" height="917" alt="Screenshot 2026-02-22 031945" src="https://github.com/user-attachments/assets/d36e188c-f87d-4d9e-9f86-0551ec5237b9" />

🧰 Tech Stack
<p> <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg" /> <img src="https://img.shields.io/badge/Streamlit-Frontend-red.svg" /> <img src="https://img.shields.io/badge/Scikit--learn-ML-green.svg" /> <img src="https://img.shields.io/badge/Pandas-Data%20Processing-blue.svg" /> <img src="https://img.shields.io/badge/Matplotlib-Visualization-orange.svg" /> </p>
📌 Table of Contents

Overview

Features

Architecture

Dataset

Project Structure

Installation

Running the App

How It Works

Dependencies

Future Enhancements

Author

🔍 Overview

Product Similarity Engine is a real-world e-commerce recommendation system inspired by Flipkart/Amazon.
It uses TF-IDF Vectorization + Cosine Similarity on product metadata to find the most relevant similar products.

This project includes:

✔ Smart search
✔ Category-based filtering
✔ Modern e-commerce UI
✔ Product cards with images, pricing, ratings
✔ Similarity bar graphs
✔ Similarity heatmap
✔ Fully interactive Streamlit web app

⚡ Designed to replicate enterprise-grade product recommendation systems in a lightweight, ML-based format.

🌟 Features
🔍 1. Smart Product Search

Instant search from product names

Optional category filter

Auto-cleaned category labels

🛒 2. E-commerce Style Product Cards

Each card displays:

Product Image

Name

Brand

Category

Discounted / Retail Price

Rating

Similarity score badge

🤖 3. ML Engine – TF-IDF + Cosine Similarity

Vectorizes product descriptions

Computes similarity with cosine similarity

Top-N product recommendations

Category-restricted matching for relevance

📊 4. Visual Insights

Includes:

Cosine Similarity horizontal bar graph

Product-to-product heatmap matrix

Dynamic analytics panel

Top categories explored

🎨 5. Streamlit Frontend

Fully modernized dark-theme UI

Responsive grid layout

Animated hero section

Clean product detail view

Custom CSS styling

🧠 Architecture
User → Streamlit UI → Search Query
         ↓
  TF-IDF Vectorizer (trained)
         ↓
Cosine Similarity Matrix
         ↓
Top-N Most Similar Products
         ↓
Visualizations + Cards + Heatmap
🗄️ Dataset

Based on Flipkart E-commerce Dataset (cleaned version).
Includes the following fields:

product_name

description

brand

discounted_price

retail_price

overall_rating

category

image_url

Preprocessing performed via:

Category normalization

Price cleaning

Image URL extraction

Description cleaning

📁 Project Structure
product-similarity-engine/
│
├── app.py                      # Streamlit frontend
├── similarity_engine.py        # ML engine (TF-IDF + similarity matrix)
├── clean_data.py               # Dataset cleaning script
│
├── data/
│   ├── flipkart_raw.csv
│   └── products_clean.csv
│
├── requirements.txt
└── README.md
🛠️ Installation
1. Clone the Repository
git clone https://github.com/riddhi-sharma10/product-similarity-engine.git
cd product-similarity-engine
2. Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Application

Run Streamlit:

streamlit run app.py

Expected output:

Local URL: http://localhost:8501

Open the URL in your browser.

🔁 How It Works

User searches for a product

Engine matches product name → retrieves full record

TF-IDF vector for selected product is compared with all others

Cosine similarity scores computed

Results filtered by category

Output visualized via:

Product cards

Bar graph

Heatmap matrix

📦 Dependencies
Package	Purpose
streamlit	Web interface
pandas	Data loading & cleaning
numpy	Numerical operations
scikit-learn	TF-IDF & cosine similarity
matplotlib	Bar graph & heatmap
re	Text cleaning
collections	Analytics counters
🚀 Future Enhancements

Image similarity using CNN embeddings

Hybrid recommender (text + price + brand + image)

Personalized recommendations

REST API backend for integration

Product clustering dashboard

Deploy on HuggingFace / Render

👤 Author — Riddhi Sharma

🎓 Computer Science Engineering · AI/ML & Web Dev
📧 riddhisharma240604@gmail.com

💼 linkedin.com/in/riddhi-sharma10
🐱 github.com/riddhi-sharma10

<p align="center"> Made with ❤️ by Riddhi Sharma · © 2025 </p>
