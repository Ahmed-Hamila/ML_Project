import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Rating data loading function for .csv files
def load_rating_data_csv(file_path: Path) -> pd.DataFrame:
    """
    Loads rating data from a CSV file into a pandas dataframe.
    :param file_path: pathlib.Path to the .csv file containing rating data. (using pathlib.path for universal compatibility between OS)
    :return: DataFrame containing the rating data.
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# Rating data loading function for .dat files (specific format handling for MovieLens 1M dataset)
def load_rating_data_dat(file_path: Path) -> pd.DataFrame:
    """
    Loads rating data from a .dat file into a pandas dataframe.
    :param file_path: pathlib.Path to the .dat file containing rating data. (using pathlib.path for universal compatibility between OS)
    :return: DataFrame containing the rating data.
    """
    try:
        # The MovieLens 1M dataset uses '::' as a delimiter and has no header
        # Defined column names based on the other dataset present in the project (32M dataset) (userId, movieId, rating, timestamp)
        data = pd.read_csv(file_path, delimiter="::", header = None, names=['userId', 'movieId', 'rating', 'timestamp'])
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# General dataset cleanup function
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing duplicates and handling missing values as well as dropping the timestamp column as it serves no purpose for us.
    :param df: DataFrame containing the rating data.
    :return: df: Cleaned DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (example: drop rows with any missing values)
    df = df.dropna()

    # Removing timestamp column if it exists as it serves no purpose in analysis
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    return df

# Rating data preprocessing function
def preprocess_rating_data(file_path: Path, file_type: str) -> pd.DataFrame:
    """
    Wrapper function to load and preprocess rating data and return the ready-to-use DataFrame.
    :param file_path: pathlib.Path to the file containing rating data. (using pathlib.path for universal compatibility between OS)
    :param file_type: Type of the file for appropriate handling ('csv' or 'dat').
    :return: df: Preprocessed DataFrame containing the rating ready-to-use data.
    """
    if file_type == 'csv':
        data = load_rating_data_csv(file_path)
    elif file_type == 'dat':
        data = load_rating_data_dat(file_path)
    else:
        print(f"Error: Unsupported file type '{file_type}'. Supported types are 'csv' and 'dat'.")
        return pd.DataFrame()

    if data.empty:
        print("No data to preprocess.")
        return data

    cleaned_data = clean_dataset(data)
    return cleaned_data

# Movie genres processing function
def process_movie_genres(df : pd.DataFrame, tag_column: str = 'genres') -> pd.DataFrame:
    """
    Processes the movie genres by splitting the genre strings into lists.
    :param df: dataframe containing movie data.
    :param tag_column: the column name containing genre tags as strings.
    :return: df: DataFrame with processed genres as lists instead of a string with '|' delimiter.
    """
    if tag_column not in df.columns:
        print(f"Error: The specified tag column '{tag_column}' does not exist in the DataFrame.")
        return df

    # Split the genres by '|' and convert to list
    df[tag_column] = df[tag_column].str.split('|')

    return df

# Movies data loading function for .csv files
def load_movies_data_csv(file_path: Path,) -> pd.DataFrame:
    """
    Loads movies data from a CSV file into a pandas dataframe.
    :param file_path: pathlib.Path to the .csv file containing movies data. (using pathlib.path for universal compatibility between OS)
    :return: df: DataFrame containing the movies data.
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        # Process movie genres if the 'genres' column exists
        if 'genres' in data.columns:
            data = process_movie_genres(data, tag_column='genres')
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# Movies data loading function for .dat files
def load_movies_data_dat(file_path: Path,) -> pd.DataFrame:
    """
    Loads movies data from a .dat file into a pandas dataframe.
    :param file_path: pathlib.Path to the .dat file containing movies data. (using pathlib.path for universal compatibility between OS)
    :return: df: DataFrame containing the movies data.
    """
    try:
        # The MovieLens 1M dataset uses '::' as a delimiter and has no header
        # Defined column names based on the other dataset present in the project (32M dataset) (movieId, title, genres)
        data = pd.read_csv(file_path, delimiter="::", header=None, names=['movieId', 'title', 'genres'])
        # Process movie genres
        data = process_movie_genres(data, tag_column='genres')
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# Movies data preprocessing function
def preprocess_movies_data(file_path: Path, file_type: str) -> pd.DataFrame:
    """
    Wrapper function to load and preprocess movies data and return the ready-to-use DataFrame.
    :param file_path: pathlib.Path to the file containing movies data. (using pathlib.path for universal compatibility between OS)
    :param file_type: Type of the file for appropriate handling ('csv' or 'dat').
    :return: df: Preprocessed DataFrame containing the movies ready-to-use data.
    """
    if file_type == 'csv':
        data = load_movies_data_csv(file_path)
    elif file_type == 'dat':
        data = load_movies_data_dat(file_path)
    else:
        print(f"Error: Unsupported file type '{file_type}'. Supported types are 'csv' and 'dat'.")
        return pd.DataFrame()

    if data.empty:
        print("No data to preprocess.")
        return data

    cleaned_data = clean_dataset(data)
    return cleaned_data

# General function to merge ratings and movies data
def merge_datasets(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the ratings and movies dataframes on the 'movieId' column.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing the movies data.
    :return: df: Merged DataFrame containing both ratings and movie information.
    """
    if 'movieId' not in ratings_df.columns or 'movieId' not in movies_df.columns:
        print("Error: 'movieId' column must be present in both DataFrames to perform the merge.")
        return pd.DataFrame()

    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')
    return merged_df

# Surprise library dataset loading function
def load_surprise_dataset(df: pd.DataFrame) -> Dataset | None:
    """
    Loads the rating data into a Surprise Dataset object for recommendation algorithms.
    :param df: DataFrame containing the rating data.
    :return: Surprise Dataset object.
    """
    try:
        reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        return data
    except Exception as e:
        print(f"An error occurred while loading the Surprise dataset: {e}")
        return None

# Train-test split function for Surprise dataset
def split_surprise_dataset(data: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple[Dataset, Dataset] | tuple[None, None]:
    """
    Splits the Surprise Dataset into training and testing sets.
    :param data: Surprise Dataset object.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: trainset, testset
    """
    try:
        trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
        return trainset, testset
    except Exception as e:
        print(f"An error occurred while splitting the Surprise dataset: {e}")
        return None, None