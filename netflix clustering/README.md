# Netflix Show Clustering

## Brief Intro

Group similar Netflix shows using unsupervised learning based on genre, rating, duration, and other metadata. Visualize the clusters to find patterns like "critically acclaimed short dramas" vs "long binge-worthy comedies."

## Tech Stack

- Python, pandas, scikit-learn (KMeans, PCA), Matplotlib/Seaborn, Streamlit

## Key highlights

- Data cleaning and encoding categorical features (genre, rating)
- Feature scaling before clustering (why KMeans breaks without it)
- Choosing optimal K using the elbow method and silhouette score
- Dimensionality reduction with PCA for visualization
<img width="1391" height="572" alt="Screenshot 2026-08-31 at 3 39 37 PM" src="https://github.com/user-attachments/assets/814257d0-3930-4c27-8a21-610af24a4e35" />
<img width="1109" height="791" alt="Screenshot 2026-08-31 at 3 40 04 PM" src="https://github.com/user-attachments/assets/c114f1e2-0070-4635-b38b-09d8bf9c59f3" />



## execution

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 02_netflix_clustering/app.py
```

## Data Sources

The app tries these sources in order:

1. **Local CSV** — Place the CSV file at `02_netflix_clustering/data/netflix_titles.csv`
2. **Kaggle** — Automatically downloads the [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset if Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
3. **Synthetic data** — Generates ~2 000 realistic rows as a fallback
