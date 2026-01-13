##    Project Overview
This project explores collaborative filtering techniques for building a movie recommendation system using the MovieLens dataset.
It focuses on two popular algorithms: Singular Value Decomposition (SVD) and K-Nearest Neighbors Baseline (KNNBaseline). 

The dataset used is the MovieLens dataset, available in multiple sizes. This project specifically utilizes the 100k and 1M ratings datasets to evaluate model performance across different data scales.
This dataset contains user ratings for movies, along with metadata such as movie titles and genres.

The project is structured into four main stages: Exploratory Data Analysis (EDA), Data Preparation, Model Training, and Evaluation & Visualization.
The main purpose is to compare the performance of these algorithms on two different dataset sizes (100k and 1M ratings) and analyze their effectiveness in predicting user preferences as well as their performance.

## 📈 Project Structure

This project is organized as follows:

### 1. Data folder
This contains the datasets used in the project.

### 2. Notebooks folder
This contains Jupyter Notebooks for each stage of the project:
*   `01_EDA.ipynb`: Exploratory Data Analysis.
*   `02_data_preparation.ipynb`: Data Preparation.
*   `03_modeling.ipynb`: Model Training.
*   `04_evaluation.ipynb`: Evaluation & Visualization.

### 3. Utils folder
This contains utility scripts for various tasks:
*   `preprocessing.py`: Functions for loading and preprocessing data.
*  `evaluation.py`: Functions for calculating performance metrics.
*   `visualization.py`: Functions for generating plots and visualizations.

### 4. Models folder 
This folder is created to store trained model files for reuse.
This allows us to avoid retraining models every time we want to evaluate or visualize results as well as saving computational resources.


## 🚀 Setup & Installation

1.  **Clone the repository** (if applicable) or navigate to the project root.
2.  **Install dependencies**:
    Ensure you have Python 3.11+ installed. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Key libraries include `pandas`, `numpy`, `scikit-learn`, `scikit-surprise`, `matplotlib`, and `jupyter`.*

## 📊 Workflow

The project is divided into four sequential stages, managed via Jupyter Notebooks:

### 1. Exploratory Data Analysis (EDA)
**Notebook:** `notebooks/01_EDA.ipynb`
*   Analyzes the distribution of ratings, genres, and user activity.
*   Visualizes insights from the raw MovieLens data.

### 2. Data Preparation
**Notebook:** `notebooks/02_data_preparation.ipynb`
*   Loads raw data using `utils/preprocessing.py`.
*   Handles different file formats (`.csv` for small, `.dat` with specific encodings for 1M).
*   Preprocesses data (merging users, movies, and ratings).
*   Splits data into Training and Testing sets.
*   Saves processed datasets as pickle files in `data/prepared-*/`.

### 3. Model Training
**Notebook:** `notebooks/03_modeling.ipynb`
*   Trains **SVD** (Singular Value Decomposition) and **KNNBaseline** models.
*   Fits models on both the 100k and 1M training sets.
*   Serializes and saves the trained models to the `models/` directory for potential reuse.

### 4. Evaluation & Visualization
**Notebook:** `notebooks/04_evaluation.ipynb`
*   Loads the pre-trained models and test datasets.
*   Computes performance metrics (e.g., RMSE, MAE) using `utils/evaluation.py`.
*   Generates comparison plots using `utils/visualization.py`.
*   Manages memory efficiently by garbage collecting large objects after use.

These notebooks are supported by utility scripts in the `utils/` directory for preprocessing, evaluation, and visualization tasks.

## 🧠 Models Used

The project compares two popular collaborative filtering algorithms:

*   **SVD (Singular Value Decomposition)**: A matrix factorization technique popularized by the Netflix Prize. It reduces the dimensionality of the user-item interaction matrix to capture latent factors.
*   **KNNBaseline**: A basic collaborative filtering algorithm that considers nearest neighbors (user-based or item-based) while accounting for baseline biases (e.g., some users rate consistently higher).

These models were used due to their effectiveness in recommendation systems with minimal preprocessing and their availability in the `scikit-surprise` library.
