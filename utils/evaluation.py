import time
from surprise import accuracy, SVD, Dataset, KNNBaseline


# Function to compute RMSE, MAE and time taken for predictions on SVD model
def evaluate_model(svd_model: SVD | KNNBaseline , testset: list) -> dict:
    """
    Evaluates the given SVD or KNNBaseline model on the provided test set.
    Computes RMSE and MAE, and measures the time taken for predictions.
    :param svd_model: Trained SVD model from Surprise library
    :param testset: Test set for evaluation
    :return: Dictionary with RMSE, MAE, and time taken
    """
    start_time = time.time()
    predictions = svd_model.test(testset)
    end_time = time.time()

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    time_taken = end_time - start_time

    return {
        'RMSE': rmse,
        'MAE': mae,
        'Time Taken (s)': time_taken
    }


# Wrapper function to evaluate 2 SVD models and a KNNBaseline model at the same time
def overall_evaluation(svd_32m_model: SVD, testset_svd_32m: list,
                       svd_1m_model: SVD, testset_svd1_m: list,
                       knn_model: KNNBaseline, testset_KNN: list) -> dict:
    """
    Evaluates two SVD models and one KNNBaseline model on their respective test sets.
    :param svd_32m_model: First trained SVD model
    :param testset_svd_32m: Test set for the first SVD model
    :param svd_1m_model: Second trained SVD model
    :param testset_svd1_m: Test set for the second SVD model
    :param knn_model: Trained KNNBaseline model
    :param testset_KNN: Test set for the KNNBaseline model
    :return: Dictionary with evaluation results for all models
    """
    results = {
        'SVD Model 1': evaluate_model(svd_32m_model, testset_svd_32m),
        'SVD Model 2': evaluate_model(svd_1m_model, testset_svd1_m),
        'KNNBaseline Model': evaluate_model(knn_model, testset_KNN)
    }
    return results