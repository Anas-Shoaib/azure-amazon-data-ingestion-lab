# Azure Data Ingestion Pipeline – Amazon Electronics Reviews (Lab02)

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
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

The file was decompressed using:
gunzip reviews_Electronics_5.json.gz

This produced the raw JSON file:
reviews_Electronics_5.json


---

## Step 3: Upload to Azure Blob Storage
A Shared Access Signature (SAS) token was generated from the Azure Portal to allow secure upload.

The dataset was uploaded using AzCopy:
azcopy copy "./reviews_Electronics_5.json" "<blob-url-with-sas>" --overwrite=true


The file was successfully stored inside the **raw** container of the data lake.

---

## Step 4: Metadata Fix
The metadata file was not valid JSON because each line was formatted as a Python dictionary (single quotes).  
This format cannot be read reliably by analytics tools.

A Python script was used to convert each line into proper JSON format.

Corrected file:
meta_Electronics_fixed.json

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

