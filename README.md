# Azure Data Engineering Pipeline – Amazon Electronics Reviews

## Labs 02 & 03 — Cloud Ingestion, Preprocessing & Lakehouse Curation

---

# Lab 02 – Azure Data Ingestion Pipeline

## Objective
The objective of this lab was to ingest a large public dataset into Azure Data Lake Storage and process it using Azure Data Factory. The lab demonstrates a complete cloud data engineering workflow including ingestion, storage, transformation, and automated processing.

---

## Step 1: Storage Account Creation
- Created an Azure Storage Account
- Enabled Hierarchical Namespace (Gen2)
- Created three containers:
  - raw
  - processed
  - curated

The storage account acts as a cloud data lake where raw and processed data are stored separately.

---

## Step 2: Dataset Acquisition
Downloaded the Amazon Electronics review dataset from the Stanford SNAP public dataset.

**Reviews dataset**
reviews_Electronics_5.json.gz

**Metadata dataset**
meta_Electronics.json.gz

The reviews dataset was downloaded directly inside the Azure VM using:
```bash
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
```

The file was decompressed using:
```bash
gunzip reviews_Electronics_5.json.gz
```

This produced the raw JSON file:
`reviews_Electronics_5.json`

---

## Step 3: Upload to Azure Blob Storage
A Shared Access Signature (SAS) token was generated from the Azure Portal to allow secure upload.

The dataset was uploaded using AzCopy:
```bash
azcopy copy "./reviews_Electronics_5.json" "<blob-url-with-sas>" --overwrite=true
```

The file was successfully stored inside the **raw** container of the data lake.

---

## Step 4: Metadata Fix
The metadata file was not valid JSON because each line was formatted as a Python dictionary (single quotes).  
This format cannot be read reliably by analytics tools.

A Python script was used to convert each line into proper JSON format.

Corrected file: `meta_Electronics_fixed.json`

The corrected metadata file was uploaded back to the **raw** container.

---

## Step 5: Azure Data Factory Pipeline
An Azure Data Factory instance was created and connected to the storage account using a Linked Service.

Datasets created:
- Raw JSON dataset (source)
- Processed Parquet dataset (sink)

A Mapping Data Flow was created to:
- Read JSON reviews
- Convert Unix timestamp to a calendar year
- Extract review year from `unixReviewTime`
- Partition data by year
- Output data into Parquet format

---

## Step 6: Pipeline Execution
A pipeline was created and executed successfully using Debug mode.

The processed data was stored in the **processed** container and partitioned by review year (approximately 1999–2014) based on the `unixReviewTime` field.

---

## Step 7: Automation
A scheduled trigger was created to demonstrate an automated ingestion workflow.  
This allows the pipeline to run automatically without manual execution in real-world scenarios.

---

## Tools and Technologies Used
- Microsoft Azure Storage Account (ADLS Gen2)
- Azure Machine Learning Compute Instance (VM)
- Azure Data Factory (Mapping Data Flow & Pipeline)
- AzCopy CLI
- Python (metadata JSON correction)

---

The following were verified during the lab:
- Storage account containers showing raw and processed zones
- Successful AzCopy upload of reviews dataset
- Azure Data Factory pipeline debug run (Succeeded)
- Partitioned Parquet output in processed container

---

## Result
The Amazon Electronics review dataset was successfully ingested, transformed, and stored in Azure Data Lake Storage in optimized Parquet format.

This lab demonstrates a complete cloud data engineering ingestion pipeline including large-scale data ingestion, transformation, partitioning, and automation using Microsoft Azure.

---

---

# Lab 03 – Data Preprocessing on Azure Databricks

## Objective
The objective of this lab was to preprocess the ingested Amazon Electronics reviews data using **Azure Databricks** and build a **Lakehouse medallion architecture** pipeline. The pipeline cleans raw data, enriches it with product metadata, and produces a curated Gold-layer dataset ready for analytics and machine learning.

---

## Technologies & Concepts Explained

### Apache Spark
Apache Spark is a distributed data processing engine that runs computations in parallel across a cluster of machines. In this lab, Spark is used via PySpark (the Python API) to read, transform, join, and write large datasets stored in Azure Data Lake. Spark's `DataFrame` API allows SQL-like operations on structured data at scale.

### Azure Databricks
Azure Databricks is a managed cloud platform built on top of Apache Spark. It provides an interactive notebook environment (similar to Jupyter) where data engineers and data scientists can write and execute Spark code. Databricks handles cluster management, scaling, and infrastructure automatically. It integrates natively with Azure Data Lake Storage Gen2 via `abfss://` paths.

### Parquet
Parquet is a columnar storage file format optimized for big data workloads. Unlike row-based formats (CSV, JSON), Parquet stores data column by column, which enables:
- Faster analytical queries (only relevant columns are read)
- Better compression ratios
- Schema enforcement and type safety

In this pipeline, Parquet is used for all Silver and Gold layer data.

### Delta Lake
Delta Lake (referenced in the Databricks architecture) is an open-source storage layer that adds ACID transactions, versioning, and schema enforcement on top of Parquet files. It enables reliable data pipelines and is commonly used in production Lakehouse architectures.

### Medallion Architecture (Bronze / Silver / Gold)
The medallion architecture organizes a data lake into three logical layers:

| Layer  | Container   | Description                                          |
|--------|-------------|------------------------------------------------------|
| Bronze | `raw/`      | Raw data as originally ingested — JSON files         |
| Silver | `processed/`| Cleaned, validated, and structured data — Parquet    |
| Gold   | `curated/`  | Use-case-ready datasets — analytics and ML features  |

Each layer progressively refines the data, making downstream consumption easier and more reliable.

---

## Architecture Overview

```
raw/                          processed/                     curated/
(Bronze)                      (Silver)                       (Gold)
  └─ reviews JSON       →       └─ clean_reviews/       →      └─ features_v1/
  └─ metadata JSON      →       └─ enriched_reviews/
```

The three Databricks notebooks each handle one stage of this pipeline:

1. `01_load_and_clean_reviews` — loads Silver Parquet data and applies cleaning rules
2. `02_enrich_with_metadata` — joins cleaned reviews with product metadata from Bronze
3. `03_write_gold_features_v1` — selects final features and writes the Gold dataset

---

## Step 1: Azure Databricks Workspace & Cluster Setup

- Created an **Azure Databricks Workspace** in the same region as the Storage Account
- Created a **Databricks cluster** with the following config:
  - Runtime: LTS (e.g., 17.3 LTS)
  - Min workers: 2, Max workers: 4
  - Auto-termination: 30 minutes (to control costs)

> **Cost tip:** Always terminate your cluster when not in use. Databricks charges by the minute while the cluster is running.

---

## Step 2: Connecting Databricks to Azure Data Lake Storage Gen2

All three notebooks begin by configuring Spark with the Storage Account key, which allows reading and writing data via `abfss://` paths:

```python
storage_account_name = "amazonXXXXXXXX"
storage_account_key = "your-key-here"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)
```

---

## Notebook 1: Load and Clean Reviews (`01_load_and_clean_reviews`)

### Purpose
Load the Silver-layer Parquet reviews data and apply cleaning and validation rules to ensure data quality.

### What the code does

**Load data:**
```python
reviews_path = "abfss://processed@amazonXXXXXX.dfs.core.windows.net/reviews/"
reviews_df = spark.read.parquet(reviews_path)
```
Reads the Parquet files produced by Lab 2's ADF pipeline. The data is partitioned by review year.

**Cleaning rules applied:**

```python
from pyspark.sql.functions import col, trim, length

clean_reviews_df = reviews_df \
    .filter(col("asin").isNotNull() & col("reviewerID").isNotNull() & col("overall").isNotNull()) \
    .filter((col("overall") >= 1) & (col("overall") <= 5)) \
    .withColumn("reviewText", trim(col("reviewText"))) \
    .filter(length(col("reviewText")) >= 10)
```

| Rule | Reason |
|------|--------|
| Drop rows with null `asin`, `reviewerID`, `overall` | These are critical identifiers — rows without them are unusable |
| Enforce rating between 1 and 5 | Ratings outside this range are invalid for the Amazon dataset |
| Trim whitespace from `reviewText` | Prevents leading/trailing spaces from causing issues in NLP or matching |
| Remove reviews shorter than 10 characters | Very short reviews are typically noise and not useful for ML |

**Output:** Cleaned data is written back to the Silver layer at `processed/clean_reviews/`.

---

## Notebook 2: Enrich Reviews with Product Metadata (`02_enrich_with_metadata`)

### Purpose
Join the cleaned reviews with product metadata (title, brand, price) from the Bronze layer to create a richer dataset.

### What the code does

**Load cleaned reviews from Silver:**
```python
clean_reviews_df = spark.read.parquet("abfss://processed@...dfs.core.windows.net/clean_reviews/")
```

**Load metadata from Bronze (raw JSON):**
```python
metadata_df = spark.read.json("abfss://raw@...dfs.core.windows.net/meta_Electronics_fixed.json")
metadata_df = metadata_df.select("asin", "title", "brand", "price")
```

Spark can read JSON and Parquet in the same pipeline, even though the formats differ. Only the four needed columns are kept to minimize memory usage.

**Left join on `asin`:**
```python
enriched_df = clean_reviews_df.join(metadata_df, on="asin", how="left")
```

A **left join** is used so that all reviews are preserved even if no matching product metadata exists. Reviews for unknown products will have `null` values in the metadata columns.

**Output:** Enriched data is written to `processed/enriched_reviews/`.

---

## Notebook 3: Write Gold Dataset (`03_write_gold_features_v1`)

### Purpose
Select the final feature set and write the curated Gold dataset to the `curated/` container.

### What the code does

**Load enriched reviews:**
```python
enriched_df = spark.read.parquet("abfss://processed@...dfs.core.windows.net/enriched_reviews/")
```

**Select final features:**
```python
features_v1_df = enriched_df.select(
    "asin", "title", "brand", "price",
    "reviewerID", "overall", "summary",
    "reviewText", "helpful", "reviewTime", "review_year"
)
```

Only the columns relevant for analytics and machine learning are kept. This reduces storage costs and speeds up downstream queries.

**Write to Gold layer:**
```python
gold_path = "abfss://curated@...dfs.core.windows.net/features_v1/"
features_v1_df.write.mode("overwrite").parquet(gold_path)
```

**Output:** Final `features_v1` dataset in the `curated/` container, ready for analytics, dashboards, or ML model training.

---

## Step 3: Orchestrating the Pipeline with Databricks Jobs

Rather than running notebooks manually, a **Databricks Job** was created to execute all three notebooks in sequence as a single automated pipeline.

### Job Configuration
- Job name: `lab3_data_preprocessing_job`
- Three tasks added in order:
  1. `load_and_clean_reviews` → runs `01_load_and_clean_reviews`
  2. `enrich_with_metadata` → runs `02_enrich_with_metadata` (depends on task 1)
  3. `write_gold_features_v1` → runs `03_write_gold_features_v1` (depends on task 2)

Task dependencies ensure the pipeline runs in order and stops if any step fails.

### Pipeline DAG
```
load_and_clean_reviews  →  enrich_with_metadata  →  write_gold_features_v1
```

### Scheduling
A daily trigger was configured on the job to demonstrate automated execution. The schedule was paused after creation to avoid unintended runs and cost accumulation.

---

## How the Notebooks Map to the ETL Process

| ETL Stage  | Notebook                        | Description                                      |
|------------|---------------------------------|--------------------------------------------------|
| Extract    | `01_load_and_clean_reviews`     | Reads raw/processed Parquet data from ADLS       |
| Transform  | `02_enrich_with_metadata`       | Cleans, validates, and joins with metadata       |
| Load       | `03_write_gold_features_v1`     | Writes the curated Gold dataset to `curated/`    |

---

## Final Data Lake State

After completing this lab, the data lake contains:

```
raw/                        ← Bronze: original JSON files
processed/
  ├── reviews/              ← Silver: Parquet from Lab 2 (partitioned by year)
  ├── clean_reviews/        ← Silver: cleaned reviews
  └── enriched_reviews/     ← Silver: reviews + product metadata
curated/
  └── features_v1/          ← Gold: final ML/analytics-ready dataset
```

---

## Tools and Technologies Used
- **Azure Databricks** — managed Spark environment, notebooks, and Jobs
- **Apache Spark (PySpark)** — distributed data processing
- **Azure Data Lake Storage Gen2 (ADLS)** — cloud data lake storage
- **Parquet** — columnar file format for efficient storage and querying
- **Medallion Architecture** — Bronze / Silver / Gold layered data organization
- **Databricks Jobs** — pipeline orchestration and scheduling

---

## Result
The Amazon Electronics reviews data was successfully preprocessed, enriched, and curated into a Gold-layer dataset using Azure Databricks and a three-stage ETL pipeline. The pipeline is orchestrated as a Databricks Job, making it reproducible and schedulable for production use.

# Lab 4 – Text Feature Engineering with Azure ML

## Overview
This lab transforms raw Amazon Electronics review text into machine-learning-ready numerical features using Azure ML Pipelines. The engineered features are registered in the Azure ML Feature Store for reuse in downstream modeling labs.

---

## Repository Structure
```
├── components/
│   ├── split_dataset/
│   │   ├── split.py
│   │   └── component.yml
│   ├── normalize_text/
│   │   ├── normalize.py
│   │   └── component.yml
│   ├── length_features/
│   │   ├── length.py
│   │   └── component.yml
│   ├── sentiment_features/
│   │   ├── sentiment.py
│   │   ├── conda.yml
│   │   └── component.yml
│   ├── tfidf_features/
│   │   ├── tfidf.py
│   │   └── component.yml
│   ├── sbert_embeddings/
│   │   ├── sbert.py
│   │   ├── conda.yml
│   │   └── component.yml
│   └── merge_features/
│       ├── merge.py
│       └── component.yml
├── pipelines/
│   └── feature_pipeline.yml
├── datastores/
│   └── curated_adls.yml
├── feature_store/
│   ├── entity_amazon_review.yml
│   ├── FeatureSetSpec.yaml
│   └── feature_set_amazon_review_text_features.yml
└── README.md
```

---

## Part 1 – Exploration, Validation, and Sampling

### Dataset
The input dataset is the curated Gold layer dataset (`features_v1`) produced in Lab 3, stored in Azure Data Lake Storage. It contains over 20 million Amazon Electronics reviews.

### Validation Steps
- Verified row count, column count, and schema
- Confirmed `reviewText` is stored as string
- Confirmed `overall` (rating) is numeric
- Confirmed entity columns `asin` and `reviewerID` are present
- Checked for missing or empty values in key columns

### Visualizations
**Rating Distribution** – Shows how reviews are distributed across star ratings (1–5). This matters for feature engineering because it reveals class imbalance, which affects how sentiment and TF-IDF features correlate with the target variable.

**Review Length Distribution** – Shows the spread of word counts and character counts across reviews. This matters because very short reviews carry less signal and may need to be filtered out, while extremely long reviews may need truncation for transformer-based models like SBERT.

### Sampling Strategy
A sample of 300,000 reviews was created using a time-stratified approach to avoid temporal drift. Language evolves over time, and a purely random sample risks over-representing reviews from a single time period. By sampling proportionally across years, the sample remains representative of the full distribution of language usage.
```python
sample_n = 300_000
sample_seed = 42
df_sampled = df.orderBy("reviewerID").limit(sample_n)
```

The sampled dataset was written to the Gold layer as `features_v1_sampled`, leaving the original dataset unchanged.

---

## Part 2 – Azure ML Feature Engineering Pipeline

### Pipeline Overview
The feature engineering pipeline reads the sampled Gold dataset, splits it to avoid data leakage, applies text normalization and feature extraction, and produces a versioned merged feature dataset registered in the Azure ML Feature Store.
```
sampled_data
     │
  split (70/15/15)
     │
  ┌──┴──────────────┐
normalize_train  normalize_val  normalize_test
     │
  ┌──┼──────────────┐
length  sentiment  tfidf  sbert
     │
  merge_all
```

---

## Components

### 1. split_dataset
**What it does:** Splits the sampled dataset into train (70%), validation (15%), and test (15%) sets using stratified random splitting.

**Why it must happen first:** Splitting before any feature fitting is critical to prevent data leakage. If TF-IDF or any scaler is fit on the full dataset before splitting, information from the validation and test sets leaks into the training process, producing artificially inflated evaluation metrics.

**Key parameters:**
- `train_ratio`: 0.7
- `val_ratio`: 0.15
- `seed`: 42 for reproducibility

---

### 2. normalize_text
**What it does:** Cleans and standardizes raw review text before feature extraction.

**Operations performed:**
- Lowercases all text
- Removes URLs using regex (`http\S+|www\S+`)
- Replaces numbers with the token `NUM` using regex (`\d+`)
- Removes punctuation
- Trims extra whitespace
- Filters out reviews shorter than 10 characters

**Why it matters:** Consistent normalization ensures that TF-IDF and SBERT features are computed on clean, uniform text. Without this step, the same word in different cases ("Great" vs "great") would be treated as different tokens, inflating the vocabulary and reducing feature quality.

---

### 3. length_features
**What it does:** Computes basic length-based features from the normalized review text.

**Features produced:**
- `review_length_words` – number of words in the review
- `review_length_chars` – number of characters in the review

**Why it matters:** Review length is a simple but informative signal. Longer reviews tend to be more detailed and often correlate with moderate ratings (3–4 stars), while very short reviews tend to be more extreme (1 or 5 stars). These features are cheap to compute and add meaningful signal to downstream models.

---

### 4. sentiment_features
**What it does:** Extracts sentiment polarity scores from the normalized review text using VADER (Valence Aware Dictionary and sEntiment Reasoner).

**Features produced:**
- `sentiment_pos` – proportion of text with positive sentiment
- `sentiment_neg` – proportion of text with negative sentiment
- `sentiment_neu` – proportion of text with neutral sentiment
- `sentiment_compound` – overall normalized polarity score (−1 = very negative, +1 = very positive)

**Why it matters:** Sentiment scores capture the emotional tone of a review, which is strongly correlated with star ratings. A model using only TF-IDF word counts may miss the overall polarity of a review; sentiment features provide a direct numerical representation of opinion.

**Library used:** `nltk.sentiment.vader` — rule-based, fast, and well-suited for short informal text like product reviews.

---

### 5. tfidf_features
**What it does:** Converts review text into a high-dimensional numerical matrix using TF-IDF (Term Frequency–Inverse Document Frequency).

**Configuration:**
- `max_features`: 5000 (top 5000 most informative terms)
- `stop_words`: English common words removed
- `ngram_range`: (1, 2) — captures unigrams and bigrams like "not good"

**Critical design decision:** The TF-IDF vectorizer is **fit only on the training split**, then applied (transformed) to the validation and test splits. This prevents vocabulary leakage from the validation and test sets into the feature representation.

**Why it matters:** TF-IDF captures which words and phrases are most informative for distinguishing between reviews. Bigrams like "not good" or "highly recommend" carry more signal than individual words alone.

---

### 6. sbert_embeddings
**What it does:** Encodes each review into a dense 384-dimensional semantic vector using Sentence-BERT (all-MiniLM-L6-v2).

**Feature produced:**
- `bert_embedding_0` through `bert_embedding_383` — 384-dimensional contextual embedding vector

**Why it matters:** Unlike TF-IDF which treats words as independent tokens, SBERT captures the semantic meaning of entire sentences. Two reviews saying "this product is amazing" and "I absolutely love this item" would have very similar SBERT embeddings even though they share no words. This deep semantic understanding significantly improves downstream model accuracy.

**Library used:** `sentence-transformers` with the `all-MiniLM-L6-v2` model — lightweight, fast, and well-suited for sentence-level encoding.

---

### 7. merge_features
**What it does:** Joins all feature datasets on the entity keys (`asin`, `reviewerID`) into a single unified feature dataset.

**Inputs merged:**
- Length features
- Sentiment features
- TF-IDF features (training split)
- SBERT embedding features

**Output:** A single Parquet file containing all engineered features, ready for Feature Store registration and downstream modeling.

---

## Running the Pipeline

### Prerequisites
- Azure ML workspace configured
- Datastore `blobkey` registered pointing to the curated container
- Data asset `amazon_electronics_features_v1_sampled` registered
- Compute cluster created

### Register All Components
```bash
az ml component create --file components/split_dataset/component.yml
az ml component create --file components/normalize_text/component.yml
az ml component create --file components/length_features/component.yml
az ml component create --file components/sentiment_features/component.yml
az ml component create --file components/tfidf_features/component.yml
az ml component create --file components/sbert_embeddings/component.yml
az ml component create --file components/merge_features/component.yml
```

### Submit the Pipeline
```bash
az ml job create --file pipelines/feature_pipeline.yml
```

### Monitor
Go to Azure ML Studio → Jobs → click the running job to see the pipeline graph.

---

## Feature Store

### Entity
The `AmazonReview` entity is defined by two index columns:
- `asin` — product identifier
- `reviewerID` — reviewer identifier

Together these uniquely identify each review in the dataset.

### Registered Feature Set
**Name:** `amazon_review_text_features`  
**Version:** 1  

| Feature | Type | Description |
|---------|------|-------------|
| review_length_words | Integer | Number of words in the review |
| review_length_chars | Integer | Number of characters in the review |
| sentiment_pos | Float | Proportion of positive sentiment |
| sentiment_neg | Float | Proportion of negative sentiment |
| sentiment_neu | Float | Proportion of neutral sentiment |
| sentiment_compound | Float | Overall polarity score (−1 to +1) |
| tfidf_0 … tfidf_4999 | Float | TF-IDF term weights |
| bert_embedding_0 … bert_embedding_383 | Float | SBERT semantic embeddings |

---

## Notes
- Storage account keys should never be committed to GitHub. Use environment variables or Azure Key Vault.
- The TF-IDF vectorizer must always be fit on training data only to prevent leakage.
- The SBERT component requires a custom conda environment with `sentence-transformers` and `torch`.
- The sentiment component requires a custom conda environment with `nltk`.
