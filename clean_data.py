import pandas as pd
import ast

# Load dataset
df = pd.read_csv("flipkart_raw.csv", encoding='latin1')

# Keep required columns
df = df[['product_name', 'description', 'image',
         'brand', 'retail_price', 'discounted_price',
         'product_rating', 'overall_rating',
         'product_category_tree']]

# Drop rows with missing values
df = df.dropna(subset=['product_name', 'description', 'image'])

# Clean image column
def clean_image(val):
    try:
        urls = ast.literal_eval(val)
        if isinstance(urls, list) and len(urls) > 0:
            return urls[0]
    except:
        pass
    return None

df['image'] = df['image'].apply(clean_image)

df = df.dropna(subset=['image'])
df = df[df['description'].str.len() > 20]

# Clean category: take only first category
df['category'] = df['product_category_tree'].apply(lambda x: x.split('>>')[0] if isinstance(x, str) else "")

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("data/products_clean.csv", index=False)

print("✔ products_clean.csv created with images + details + category!")