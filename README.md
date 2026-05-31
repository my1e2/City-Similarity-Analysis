# Module 3 Assignment — World Cities Cosine Similarity Engine

> Course: Exploratory Data Analysis  
> Module: 3  
> Dataset: World Cities Database (simplemaps.com)  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
   - [Source and Description](#source-and-description)
   - [Columns Used](#columns-used)
3. [Requirements](#requirements)
4. [Installation and Setup](#installation-and-setup)
5. [How to Run](#how-to-run)
6. [Data Pipeline](#data-pipeline)
   - [Loading and Filtering](#loading-and-filtering)
   - [Min-Max Normalization](#min-max-normalization)
7. [Similarity Engine](#similarity-engine)
   - [Function Design](#function-design)
   - [Cosine Similarity Explained](#cosine-similarity-explained)
   - [Within-Country Constraint](#within-country-constraint)
8. [Visualizations](#visualizations)
9. [Query Cities](#query-cities)
10. [Key Findings](#key-findings)
11. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

This module assignment builds a city similarity search engine using cosine similarity over a normalized feature space of geographic coordinates and population size. Given a query city, the engine identifies the ten most similar cities within the same country based on how closely their latitude, longitude, and population align in a normalized vector space. A horizontal bar chart visualizing similarity scores is produced for each query. Five cities are queried: New York, Paris, Tokyo, Mumbai, and Sydney.

---

## Dataset

### Source and Description

**File:** `worldcities.csv`  
**Source:** simplemaps.com World Cities Database  
**URL:** [https://simplemaps.com/data/world-cities](https://simplemaps.com/data/world-cities)  
**Last refreshed:** May 11, 2025  
**Coverage:** Over 4 million unique cities and towns globally (approximately 48,000 in the basic database)

The database is built from authoritative sources including the National Geospatial-Intelligence Agency (NGA), the US Geological Survey, the US Census Bureau, and NASA. It contains one entry per city with consistent formatting across all records.

---

### Columns Used

The full dataset is loaded and then reduced to only the five columns needed for this analysis:

| Column       | Type   | Description                                   |
|--------------|--------|-----------------------------------------------|
| `city`       | string | City name                                     |
| `country`    | string | Country the city belongs to                   |
| `lat`        | float  | Latitude coordinate                           |
| `lng`        | float  | Longitude coordinate                          |
| `population` | float  | City population                               |

Rows with missing values in `population`, `lat`, or `lng` are dropped. Cities with a population of zero or less are excluded.

---

## Requirements

- Python 3.x

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

| Package                              | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
| `pandas`                             | Data loading, filtering, and column management       |
| `numpy`                              | Array operations                                     |
| `sklearn.preprocessing.MinMaxScaler`| Normalizing lat, lng, and population to [0, 1]       |
| `sklearn.metrics.pairwise.cosine_similarity` | Computing similarity scores between city vectors |
| `matplotlib`                         | Figure management                                    |
| `seaborn`                            | Horizontal bar chart per query city                  |

---

## Installation and Setup

1. Download `worldcities.csv` from [https://simplemaps.com/data/world-cities](https://simplemaps.com/data/world-cities) and place it in the same directory as the notebook.
2. Open `Module3Assignment.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom.

---

## How to Run

Open the notebook and run the single code cell. The script loads the data, normalizes the three feature columns, defines the `similar_cities()` function, and then iterates over the five query cities. For each query, the top 10 most similar cities are printed to the console and a bar chart is displayed inline.

---

## Data Pipeline

### Loading and Filtering

```python
df = pd.read_csv("worldcities.csv")

cols_needed = ["city", "country", "lat", "lng", "population"]
cols = [c for c in cols_needed if c in available_cols]

df = df[cols].dropna(subset=["population", "lat", "lng"])
df = df[df["population"] > 0]
```

The `[c for c in cols_needed if c in available_cols]` pattern is a defensive selection that ensures the code does not break if the CSV schema changes — only columns that actually exist in the file are selected. The final dataset covers a subset of the global city database filtered to records with complete geographic and population data.

---

### Min-Max Normalization

Before computing cosine similarity, the three numeric features are normalized to the range [0, 1] using `MinMaxScaler`. This is critical because the raw scales of the three features differ dramatically:

- `lat` ranges approximately from -90 to +90
- `lng` ranges approximately from -180 to +180
- `population` ranges from hundreds to tens of millions

Without normalization, population would dominate the similarity calculation simply by virtue of its larger numeric magnitude, causing the similarity engine to find cities that are nearby in population regardless of geographic position. Min-Max scaling ensures each feature contributes equally to the final similarity score:

```python
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[["lat", "lng", "population"]] = scaler.fit_transform(df[["lat", "lng", "population"]])
```

A copy of the original DataFrame is maintained (`df.copy()`) so that the unscaled values remain accessible for display in the printed output.

---

## Similarity Engine

### Function Design

The `similar_cities(query_city, df, df_scaled, top_k=10)` function:

1. Validates that the query city exists in the dataset
2. Retrieves the original row for the query city from `df` to access its country
3. Retrieves the scaled feature vector for the query city from `df_scaled`
4. Filters both DataFrames to only rows in the same country as the query city
5. Computes cosine similarity between the query vector and all country-subset vectors
6. Adds the similarity scores as a new column to the subset copy
7. Removes the query city itself from the results
8. Returns the top `top_k` rows sorted by similarity descending

---

### Cosine Similarity Explained

Cosine similarity measures the angle between two vectors in the feature space rather than the Euclidean distance between them. It produces a score between 0 and 1 where 1 means the vectors point in exactly the same direction (maximally similar) and 0 means they are orthogonal (no similarity in this space).

For this analysis, two cities are similar if their normalized latitude, longitude, and population values form vectors pointing in the same direction — meaning they occupy a similar position (both geographically and in terms of population size) within the scaled feature space:

```python
cosine_sim = cosine_similarity(query_value, subset_scaled[["lat", "lng", "population"]])[0]
```

The `[0]` index extracts the first (and only) row of the similarity matrix, yielding a 1D array of similarity scores — one score per city in the country subset.

---

### Within-Country Constraint

The similarity search is deliberately constrained to cities within the same country as the query city:

```python
subset = df[df["country"] == query_row["country"]]
subset_scaled = df_scaled.loc[subset.index]
```

This design choice makes the results practically meaningful: rather than returning cities with similar populations and coordinates globally (which would find cities on different continents that happen to share similar coordinates), it focuses the comparison on domestic context — for instance, finding the US cities most similar to New York in terms of location and size within the US.

A guard is also included to handle countries with only one city in the dataset, which would make comparison impossible.

---

## Visualizations

For each query city, a horizontal bar chart is produced:

**Type:** `seaborn.barplot` with `hue` set to city name (Spectral palette)  
**Figure size:** 8x4  
**X-axis:** Cosine similarity score  
**Y-axis:** City name  
**Title:** "Top {k} Similar Cities to {query_city} ({country})"

The `hue` parameter is used to enable per-bar coloring from the Spectral palette. Bars are sorted by similarity score in descending order so the most similar city appears at the top.

---

## Query Cities

The following five cities are queried in sequence:

| Query City | Country       | Comparison Scope                              |
|------------|---------------|-----------------------------------------------|
| New York   | United States | Top 10 most similar US cities                 |
| Paris      | France        | Top 10 most similar French cities             |
| Tokyo      | Japan         | Top 10 most similar Japanese cities           |
| Mumbai     | India         | Top 10 most similar Indian cities             |
| Sydney     | Australia     | Top 10 most similar Australian cities         |

---

## Key Findings

- For large metropolitan areas like New York and Tokyo, the most similar cities tend to be other large cities in the same country with comparable populations and geographic positions. Chicago and Los Angeles consistently appear as highly similar to New York, for example.
- For cities in countries with fewer very large urban centers (such as Sydney in Australia), the similarity scores for the top results may be noticeably lower than for countries with many large cities, reflecting the fact that Australia has fewer cities closely matching Sydney's scale.
- Geographic proximity and population similarity trade off depending on a city's position within the country. Cities near the same coast or latitude as the query city may rank high even if their population differs somewhat.
- The within-country constraint produces results that are more interpretable than a global search would, since global results would often match cities on different continents that share coordinates by coincidence.

---

## Limitations and Future Work

- The similarity engine uses only three features (lat, lng, population). Adding additional features such as elevation, GDP per capita, or climate zone would produce richer and more nuanced similarity results.
- Population data in the simplemaps database varies in update frequency by country. For some smaller or less-documented cities, population figures may be outdated or estimated.
- Cosine similarity on geographic coordinates has a known limitation: it does not account for the spherical geometry of the Earth. Two cities at the same latitude but on opposite sides of the globe could receive high similarity scores due to the symmetry of the coordinate space. Haversine distance would be a more geographically accurate measure.
- The engine always compares within-country. A cross-country search mode would be a useful extension for finding internationally analogous cities (e.g., finding the "Sydney equivalent" in the United States or Europe).
- The `similar_cities()` function uses `df[df["city"] == query_city].iloc[0]` to resolve the query city, which selects the first match if the city name appears multiple times in the dataset (e.g., multiple cities named "Paris" in different countries). A country argument should be added to the function signature for disambiguation.
