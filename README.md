# Wine Quality Prediction Model

A demo MLOps project demonstrating data versioning with DVC (Data Version Control) and experiment tracking with MLflow for predicting wine quality using Random Forest Regression.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [How DVC Works](#how-dvc-works)
- [Real-World Scenario: Updating the Dataset](#real-world-scenario-updating-the-dataset)
- [MLflow Production Architecture](#mlflow-production-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [DVC Remote Setup (S3)](#dvc-remote-setup-s3)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project predicts wine quality based on various physicochemical properties using a Random Forest Regressor. It demonstrates modern MLOps practices including:

- **Data Versioning**: Using DVC to track and version the wine dataset
- **Experiment Tracking**: Using MLflow to log parameters, metrics, and models
- **Remote Storage**: Storing large datasets in S3 while keeping Git lightweight
- **Reproducibility**: Ensuring consistent results across different environments

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Wine Prediction Project                           │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌─────────────────────┐
│   Git Repo   │         │  DVC Remote  │         │  MLflow (K8s Prod)  │
│              │         │   (AWS S3)   │         │                     │
│  - Code      │         │              │         │  ┌───────────────┐  │
│  - .dvc files│◄───────►│ - wine.csv   │         │  │ MLflow Server │  │
│  - train.py  │         │ - Models     │         │  │  (K8s Pod)    │  │
│  - utils.py  │         │ - Artifacts  │         │  └───────┬───────┘  │
└──────────────┘         └──────────────┘         │          │          │
       │                        │                 │          ▼          │
       │                        │                 │  ┌───────────────┐  │
       └────────────────────────┼─────────────────┼► │  AWS RDS      │  │
                                │                 │  │  PostgreSQL   │  │
                                │                 │  │  (Metadata)   │  │
                                │                 │  └───────────────┘  │
                                │                 └─────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │  Local/CI Environment│
                    │                      │
                    │  1. dvc pull         │
                    │  2. python train.py  │
                    │  3. dvc push         │
                    └──────────────────────┘
```

### Production Architecture Components

**Data Versioning (DVC + S3)**
- **Git Repository**: Tracks code, configurations, and `.dvc` pointer files
- **AWS S3**: Remote storage for datasets, models, and large artifacts
- **DVC**: Manages data versioning and synchronization between local and S3

**Experiment Tracking (MLflow Production)**
- **MLflow Server**: Hosted as a containerized service in **Kubernetes cluster**
- **AWS RDS PostgreSQL**: Backend database storing experiment metadata, parameters, metrics, and model registry
- **Artifact Store**: Uses S3 for storing model artifacts and files
- **High Availability**: Kubernetes provides auto-scaling, load balancing, and fault tolerance

### Workflow

1. **Git** tracks code, configurations, and `.dvc` files (pointers to data)
2. **DVC** manages large data files and stores them in S3
3. **MLflow** (production) logs experiments, parameters, and metrics to Kubernetes-hosted server
4. **AWS RDS PostgreSQL** stores all MLflow metadata persistently
5. **Train script** loads data from S3 via DVC, trains model, and logs to MLflow production server

---

## How DVC Works

### What is DVC?

DVC (Data Version Control) is a version control system for data and ML models. It works alongside Git to manage large files efficiently.

### Key Concepts

```
Traditional Git (Problem):
┌─────────────────────────────────────────────┐
│ Git Repo (bloated)                          │
│                                             │
│  ✗ wine.csv (10 MB)                         │
│  ✗ wine_v2.csv (11 MB)                      │
│  ✗ model.pkl (50 MB)                        │
│  ✗ Large files make Git slow                │
└─────────────────────────────────────────────┘

DVC Solution:
┌──────────────────────┐         ┌─────────────────────────┐
│ Git Repo (light)     │         │ DVC Remote (S3)         │
│                      │         │                         │
│  ✓ wine.csv.dvc (89B)│────────►│  wine.csv (10 MB)       │
│  ✓ train.py          │         │  [hash: 5b05298d...]    │
│  ✓ utils.py          │         │                         │
│  Fast & efficient    │         │  Stores actual data     │
└──────────────────────┘         └─────────────────────────┘
```

### DVC Workflow

```
1. Add data to DVC:
   $ dvc add data/wine.csv

   Creates:
   - data/wine.csv.dvc (pointer file → tracked by Git)
   - Adds wine.csv to .gitignore (actual file → not tracked by Git)

2. Configure remote storage:
   $ dvc remote add -d myremote s3://mybucket/path

   - Tells DVC where to store actual data
   - Config saved in .dvc/config

3. Push data to remote:
   $ dvc push

   - Uploads wine.csv to S3
   - Local file → S3 storage

4. Pull data (new machine):
   $ dvc pull

   - Downloads wine.csv from S3
   - S3 storage → Local file
```

### Benefits

- **Git stays lightweight**: Only `.dvc` files (tiny pointers) go into Git
- **Data versioning**: Track dataset changes across time
- **Storage efficiency**: Store large files in S3, not in Git
- **Reproducibility**: Anyone can pull exact dataset versions
- **Collaboration**: Team members share data through S3

---

## Real-World Scenario: Updating the Dataset

Let's walk through what happens when a data scientist updates the `wine.csv` file.

### Before the Update

```
Git Repo                    DVC Remote (S3)
├── wine.csv.dvc           wine.csv
│   md5: 5b05298d...  ────► [hash: 5b05298d...]
│   size: 10948            [size: 10948 bytes]
```

### The Update Process

#### Step 1: Data Scientist Modifies wine.csv

The data scientist adds new wine samples or fixes data quality issues in `wine.csv`.

```bash
# File is now different locally
# New size: 12,500 bytes (was 10,948 bytes)
```

#### Step 2: Check DVC Status

```bash
$ dvc status

# Output:
data/wine.csv.dvc:
    changed outs:
        modified:           data/wine.csv
```

DVC detects that the file content has changed by comparing hashes!

#### Step 3: Add Updated File to DVC

```bash
$ dvc add data/wine.csv
```

**What happens internally:**

```
1. DVC calculates new MD5 hash of wine.csv
   Old hash: 5b05298d735aba05e3707b0c5784a443
   New hash: 7c21a89f812bca31d4e9c1a2b3f4d567  ← NEW!

2. Updates wine.csv.dvc file:
   outs:
   - md5: 7c21a89f812bca31d4e9c1a2b3f4d567  ← CHANGED
     size: 12500                             ← CHANGED
     hash: md5
     path: wine.csv

3. Stores new version in local DVC cache:
   .dvc/cache/7c/21a89f812bca31d4e9c1a2b3f4d567
```

#### Step 4: Push New Version to S3

```bash
$ dvc push
```

**What happens in S3:**

```
S3 Storage (Before):
s3://bucket/wine-project/
└── 5b/05298d735aba05e3707b0c5784a443  ← old version

S3 Storage (After):
s3://bucket/wine-project/
├── 5b/05298d735aba05e3707b0c5784a443  ← old version (kept!)
└── 7c/21a89f812bca31d4e9c1a2b3f4d567  ← NEW version

Both versions are stored! DVC maintains full history.
```

#### Step 5: Commit Updated Metadata to Git

```bash
$ git add data/wine.csv.dvc
$ git commit -m "Update wine dataset with 200 new samples"
$ git push
```

**Git commit history:**

```
Commit 2 (NEW):
  wine.csv.dvc → md5: 7c21a89f... (12,500 bytes)

Commit 1 (OLD):
  wine.csv.dvc → md5: 5b05298d... (10,948 bytes)
```

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ DATA SCIENTIST UPDATES wine.csv                                 │
└─────────────────────────────────────────────────────────────────┘

1. MODIFY FILE
   wine.csv (local) is edited
   ↓

2. DVC ADD
   $ dvc add data/wine.csv
   ↓
   ┌────────────────────────────────────┐
   │ DVC does:                          │
   │ • Calculate new MD5: 7c21a89f...   │
   │ • Update wine.csv.dvc file         │
   │ • Cache file in .dvc/cache/        │
   └────────────────────────────────────┘
   ↓

3. DVC PUSH
   $ dvc push
   ↓
   ┌────────────────────────────────────┐
   │ Upload to S3:                      │
   │ s3://bucket/.../7c21a89f...        │
   │ (old version 5b05298d... remains)  │
   └────────────────────────────────────┘
   ↓

4. GIT COMMIT & PUSH
   $ git add wine.csv.dvc
   $ git commit -m "Update dataset"
   $ git push
   ↓
   ┌────────────────────────────────────┐
   │ Git stores:                        │
   │ wine.csv.dvc with new hash         │
   │ (metadata only, ~89 bytes)         │
   └────────────────────────────────────┘
```

### Version History Timeline

```
═══════════════════════════════════════════════════════════════════

Day 1:  wine.csv (v1)  →  Git: wine.csv.dvc  →  S3: 5b05.../wine.csv
        10,948 bytes      (md5: 5b05...)         (10,948 bytes)

Day 5:  wine.csv (v2)  →  Git: wine.csv.dvc  →  S3: 7c21.../wine.csv
        12,500 bytes      (md5: 7c21...)         5b05.../wine.csv ✓
        (updated)         (updated)              (both versions!)

═══════════════════════════════════════════════════════════════════
```

### Retrieving Old Versions

Any team member can retrieve the old version of the dataset:

```bash
# 1. Checkout the old Git commit
$ git checkout <commit-hash-from-day-1>

# 2. Pull the corresponding dataset version
$ dvc checkout
# or
$ dvc pull

# wine.csv is now the old version (10,948 bytes)!
```

### Key Takeaways

| Aspect | What Happens |
|--------|-------------|
| **DVC detects changes** | Compares file MD5 hash with hash in `.dvc` file |
| **Metadata file updated** | `wine.csv.dvc` gets new hash and size |
| **S3 storage** | Both old and new versions stored (immutable history) |
| **Git storage** | Only stores tiny `.dvc` files (~89 bytes) |
| **Version history** | Full history maintained across Git and S3 |
| **Reproducibility** | Anyone can checkout and retrieve any version |
| **Team collaboration** | Team members run `dvc pull` to get latest version |

### Summary

```
┌──────────────────┬────────────────────┬─────────────────────┐
│   What Changed   │   Where Stored     │   Version Control   │
├──────────────────┼────────────────────┼─────────────────────┤
│ wine.csv         │ S3 (both versions) │ DVC (by hash)       │
│ wine.csv.dvc     │ Git                │ Git (by commit)     │
│ .dvc/config      │ Git                │ Git (by commit)     │
└──────────────────┴────────────────────┴─────────────────────┘
```

This workflow ensures:
- **Data versioning** without bloating Git
- **Complete history** of all dataset versions
- **Team synchronization** through S3
- **Reproducibility** across time and team members

---

## MLflow Production Architecture

### Overview

This project uses **production-grade MLflow** for experiment tracking, hosted in a **Kubernetes cluster** with **AWS RDS PostgreSQL** as the backend database. This setup provides enterprise-level reliability, scalability, and centralized experiment management.

### Architecture Components

```
┌──────────────────────────────────────────────────────────────────┐
│                    MLflow Production Stack                       │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Kubernetes Cluster (AWS EKS)                   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              MLflow Tracking Server                    │     │ 
│  │              (Deployment + Service)                    │     │
│  │                                                        │     │
│  │  • Receives experiment logs from training jobs         │     │
│  │  • Exposes REST API for tracking                       │     │
│  │  • Serves MLflow UI (web interface)                    │     │
│  │  • Manages model registry operations                   │     │
│  └────────────────┬───────────────────────┬───────────────┘     │
│                   │                       │                     │
└───────────────────┼───────────────────────┼─────────────────────┘
                    │                       │
                    ▼                       ▼
        ┌────────────────────┐   ┌──────────────────────┐
        │   AWS RDS          │   │   AWS S3             │
        │   PostgreSQL       │   │   Artifact Store     │
        │                    │   │                      │
        │ • Experiments      │   │ • Model files        │
        │ • Runs             │   │ • Plots/images       │
        │ • Parameters       │   │ • Datasets           │
        │ • Metrics          │   │ • Custom artifacts   │
        │ • Tags             │   │                      │
        │ • Model Registry   │   └──────────────────────┘
        └────────────────────┘
                    ▲
                    │
        ┌───────────┴───────────┐
        │  Training Jobs        │
        │  (Local/CI/Airflow)   │
        │                       │
        │  python train.py      │
        │  --experiment=wine    │
        └───────────────────────┘
```

### Why Kubernetes + RDS PostgreSQL?

**Kubernetes Benefits:**
- **High Availability**: Auto-restart failed pods, maintain uptime
- **Scalability**: Handle multiple concurrent experiments from different teams
- **Load Balancing**: Distribute traffic across multiple MLflow server replicas
- **Resource Management**: Control CPU/memory allocation for the tracking server
- **Rolling Updates**: Zero-downtime deployments when updating MLflow

**AWS RDS PostgreSQL Benefits:**
- **Managed Service**: Automated backups, patching, and maintenance
- **Durability**: Multi-AZ deployment for disaster recovery
- **Performance**: Optimized for high transaction workloads (experiment logging)
- **Scalability**: Vertical and horizontal scaling as experiment volume grows
- **Security**: VPC isolation, encryption at rest and in transit

### How It Works

#### 1. Training Job Execution

```python
import mlflow
import os

# Point to production MLflow server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# Example: http://mlflow-service.ml-ops.svc.cluster.local:5000

mlflow.set_experiment("wine-prediction")

with mlflow.start_run(run_name="experiment-1"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", 0.65)
    mlflow.sklearn.log_model(model, "model")
```

#### 2. Data Flow

```
Training Script
      │
      │ (1) Send experiment data via REST API
      ▼
MLflow Server (K8s Pod)
      │
      ├─► (2) Store metadata → AWS RDS PostgreSQL
      │       (params, metrics, tags, run info)
      │
      └─► (3) Store artifacts → AWS S3
          (models, plots, files)
```

#### 3. Accessing Experiments

**Via Web UI:**
```bash
# MLflow UI is accessible via Kubernetes Ingress/LoadBalancer
# Example: https://mlflow.company.com
```

**Via API:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow-k8s-url:5000")
experiments = client.list_experiments()
runs = client.search_runs(experiment_ids=["1"])
```

### Production Setup (Environment Variables)

```bash
# Set MLflow tracking URI to Kubernetes service
export MLFLOW_TRACKING_URI="http://mlflow-service.ml-ops.svc.cluster.local:5000"

# Or via external load balancer
export MLFLOW_TRACKING_URI="https://mlflow.company.com"

# Run training script
python train.py --experiment wine-prediction
```

### Database Schema in RDS PostgreSQL

The PostgreSQL database stores the following MLflow metadata:

| Table | Description |
|-------|-------------|
| **experiments** | Experiment definitions and metadata |
| **runs** | Individual experiment runs |
| **params** | Hyperparameters logged for each run |
| **metrics** | Performance metrics (RMSE, R², etc.) |
| **tags** | Custom tags for organizing experiments |
| **model_versions** | Model registry information |

### Benefits of This Architecture

1. **Centralized Tracking**: All team members log to the same MLflow server
2. **Persistent Storage**: Experiments survive local machine failures
3. **Collaboration**: Share experiments and models across the team
4. **Scalability**: Handle hundreds of concurrent experiment runs
5. **Security**: VPC-isolated RDS and role-based access control
6. **Audit Trail**: Complete history of all experiments in PostgreSQL
7. **Integration**: Works seamlessly with CI/CD pipelines (Jenkins, GitLab CI, Airflow)

### Local Development vs Production

| Aspect | Local Development | Production (K8s + RDS) |
|--------|-------------------|------------------------|
| **MLflow Server** | `mlflow ui` on localhost | Kubernetes Deployment |
| **Database** | SQLite file | AWS RDS PostgreSQL |
| **Artifacts** | Local filesystem | AWS S3 |
| **Access** | http://localhost:7006 | https://mlflow.company.com |
| **Scalability** | Single user | Multi-user, concurrent |
| **Persistence** | Local only | Centralized, durable |

---

## Prerequisites

- Python 3.8+
- Git
- AWS account with S3 bucket (for remote storage)
- AWS CLI configured with credentials

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Wine-Prediction-Model
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install DVC with S3 support

```bash
pip install dvc[s3]
```

---

## DVC Remote Setup (S3)

### Step 1: Create S3 Bucket

```bash
# Create an S3 bucket (replace with your bucket name)
aws s3 mb s3://your-wine-dataset-bucket
```

### Step 2: Configure DVC Remote

```bash
# Add S3 as the default DVC remote storage
dvc remote add -d winesetremote s3://your-wine-dataset-bucket/wine-project

# This updates .dvc/config with remote configuration
```

**Replace `s3://your-wine-dataset-bucket/wine-project` with your actual S3 bucket path.**

**Example `.dvc/config` file after this step:**

```ini
[core]
    remote = winesetremote

['remote "winesetremote"']
    url = s3://your-wine-dataset-bucket/wine-project
```

This configuration file is stored at `.dvc/config` and tells DVC:
- **`core.remote`**: The default remote storage is `winesetremote`
- **`remote "winesetremote".url`**: The S3 bucket URL where data will be stored

### Step 3: Configure AWS Credentials (Optional)

If not using default AWS CLI credentials:

```bash
dvc remote modify winesetremote access_key_id 'YOUR_ACCESS_KEY'
dvc remote modify winesetremote secret_access_key 'YOUR_SECRET_KEY'
```

**Example `.dvc/config` with AWS credentials:**

```ini
[core]
    remote = winesetremote

['remote "winesetremote"']
    url = s3://your-wine-dataset-bucket/wine-project
    access_key_id = AKIAIOSFODNN7EXAMPLE
    secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

Or use AWS profiles:

```bash
dvc remote modify winesetremote profile myprofile
```

**Example `.dvc/config` with AWS profile:**

```ini
[core]
    remote = winesetremote

['remote "winesetremote"']
    url = s3://your-wine-dataset-bucket/wine-project
    profile = myprofile
```

**Example `.dvc/config` with region specification:**

```ini
[core]
    remote = winesetremote

['remote "winesetremote"']
    url = s3://your-wine-dataset-bucket/wine-project
    region = us-east-1
```

### Step 4: Push Data to S3

```bash
# Push your tracked data to the remote storage
dvc push
```

This uploads `data/wine.csv` to your S3 bucket.

### Step 5: Verify Remote Configuration

```bash
# Check your DVC configuration
dvc remote list

# Should output:
# winesetremote  s3://your-wine-dataset-bucket/wine-project

# View the actual config file
cat .dvc/config
```

You can also manually view/edit `.dvc/config` to verify the configuration.

### Step 6: Commit DVC Configuration

```bash
git add .dvc/config
git commit -m "Configure DVC remote storage on S3"
git push
```

---

## Usage

### Pulling Data (First Time Setup)

When cloning this repository on a new machine:

```bash
# Pull the dataset from S3
dvc pull
```

This downloads `data/wine.csv` from S3 to your local machine.

### Training the Model

```bash
# Basic training with default parameters
python train.py --csv data/wine.csv

# Custom parameters
python train.py \
  --csv data/wine.csv \
  --target quality \
  --n-estimators 100 \
  --max-depth 10 \
  --test-size 0.2 \
  --experiment wine-prediction \
  --run my-experiment-run
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv` | Path to wine CSV dataset | `data/wine_sample.csv` |
| `--target` | Target column name | `quality` |
| `--experiment` | MLflow experiment name | `wine-prediction` |
| `--run` | MLflow run name | `run-2` |
| `--n-estimators` | Number of trees in Random Forest | `50` |
| `--max-depth` | Maximum depth of trees | `5` |
| `--test-size` | Test set fraction | `0.2` |
| `--random-state` | Random seed for reproducibility | `42` |

### MLflow Tracking

**Production Environment:**

Access the MLflow UI via the Kubernetes-hosted server:

```bash
# Access via browser (URL provided by your DevOps team)
# Example: https://mlflow.company.com
```

The tracking URI is set via environment variable:

```bash
export MLFLOW_TRACKING_URI="https://mlflow.company.com"
# or internal K8s service URL
export MLFLOW_TRACKING_URI="http://mlflow-service.ml-ops.svc.cluster.local:5000"
```

**Local Development:**

For local development and testing, you can run MLflow locally:

```bash
# Start local MLflow UI
mlflow ui --port 7006

# Then open http://localhost:7006 in your browser
```

Set local tracking URI:

```bash
export MLFLOW_TRACKING_URI="http://localhost:7006"
```

The `train.py` script automatically uses the `MLFLOW_TRACKING_URI` environment variable (defaults to `http://localhost:7006` if not set).

### Updating the Dataset

If you modify `data/wine.csv`:

```bash
# Track the new version
dvc add data/wine.csv

# Push to S3
dvc push

# Commit the updated .dvc file
git add data/wine.csv.dvc
git commit -m "Update wine dataset"
git push
```

---

## Project Structure

```
Wine-Prediction-Model/
│
├── data/
│   ├── wine.csv              # Actual dataset (not in Git)
│   ├── wine.csv.dvc          # DVC pointer file (in Git)
│   └── .gitignore            # Ignores wine.csv
│
├── .dvc/
│   ├── config                # DVC configuration (remote storage)
│   ├── .gitignore            # DVC internal files
│   └── cache/                # Local DVC cache
│
├── train.py                  # Main training script with MLflow
├── utils.py                  # Utility functions (data loading)
├── requirements.txt          # Python dependencies
├── .dvcignore                # Files to ignore in DVC
├── .gitignore                # Files to ignore in Git
└── README.md                 # This file
```

### File Roles

- **`wine.csv.dvc`**: Pointer file containing MD5 hash of `wine.csv`. Tracked by Git.
- **`wine.csv`**: Actual dataset file. Tracked by DVC, stored in S3.
- **`.dvc/config`**: DVC configuration including remote storage settings.
- **`train.py`**: Trains RandomForestRegressor and logs to MLflow.
- **`utils.py`**: Helper functions for data preprocessing.

---

## Technologies Used

### Core ML Stack

| Technology | Purpose |
|------------|---------|
| **scikit-learn** | Machine learning (Random Forest Regressor) |
| **pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |

### MLOps Infrastructure

| Technology | Purpose |
|------------|---------|
| **DVC** | Data version control and pipeline management |
| **Git** | Code version control |
| **AWS S3** | Remote storage for datasets and MLflow artifacts |
| **MLflow** | Experiment tracking, model registry, and versioning |
| **Kubernetes (K8s)** | Container orchestration for MLflow server |
| **AWS RDS PostgreSQL** | Backend database for MLflow metadata storage |
| **Docker** | Containerization of MLflow server |

### Infrastructure Components

```
┌─────────────────────────────────────────────────────────┐
│                   Technology Stack                      │
└─────────────────────────────────────────────────────────┘

Data Layer:
  • AWS S3        → Datasets, models, artifacts
  • DVC           → Version control for data

Compute Layer:
  • Python 3.8+   → Training scripts
  • scikit-learn  → ML algorithms

Tracking Layer:
  • MLflow        → Experiment tracking
  • Kubernetes    → MLflow hosting
  • PostgreSQL    → Metadata storage
  • AWS RDS       → Managed database

Version Control:
  • Git           → Code versioning
  • DVC           → Data versioning
```

---

## Common DVC Commands

```bash
# Check DVC status
dvc status

# Pull data from remote
dvc pull

# Push data to remote
dvc push

# List configured remotes
dvc remote list

# Get file from specific version
dvc checkout

# Compare DVC tracked files
dvc diff
```

---

## Troubleshooting

### Issue: `dvc pull` fails

**Solution**: Verify AWS credentials and S3 bucket permissions.

```bash
aws s3 ls s3://your-wine-dataset-bucket/
```

### Issue: MLflow tracking URI not found

**Solution**: Set the MLflow tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://localhost:7006
```

Or start MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 7006
```

### Issue: Dataset not found

**Solution**: Pull the dataset from DVC remote:

```bash
dvc pull
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

---

## License

This is a demo project for educational purposes.

---

## Contact

For questions or issues, please open an issue in the repository.
