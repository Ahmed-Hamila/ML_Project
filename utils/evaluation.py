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