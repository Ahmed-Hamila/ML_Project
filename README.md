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

## 🧠 Models Used

The project compares two popular collaborative filtering algorithms:

*   **SVD (Singular Value Decomposition)**: A matrix factorization technique popularized by the Netflix Prize. It reduces the dimensionality of the user-item interaction matrix to capture latent factors.
*   **KNNBaseline**: A basic collaborative filtering algorithm that considers nearest neighbors (user-based or item-based) while accounting for baseline biases (e.g., some users rate consistently higher).

These users were used due to their effectiveness in recommendation systems with minimal preprocessing and their availability in the `scikit-surprise` library.
