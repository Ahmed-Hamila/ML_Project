import pandas as pd
from pathlib import Path

# Rating data loading function for .csv files
def load_rating_data_csv(file_path: Path) -> pd.DataFrame:
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
def clean_rating_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

    cleaned_data = clean_rating_dataset(data)
    return cleaned_data

# Movie genres processing function
def process_movie_genres(df : pd.DataFrame, tag_column: str = 'genres') -> pd.DataFrame:
    if tag_column not in df.columns:
        print(f"Error: The specified tag column '{tag_column}' does not exist in the DataFrame.")
        return df

    # Split the genres by '|' and convert to list
    df[tag_column] = df[tag_column].str.split('|')

    return df

# Movies data loading function for .csv files
def load_movies_data_csv(file_path: Path,) -> pd.DataFrame:
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


# General movies dataset cleanup function
def clean_movies_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (example: drop rows with any missing values)
    df = df.dropna()

    return df

# Movies data preprocessing function
def preprocess_movies_data(file_path: Path, file_type: str) -> pd.DataFrame:
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

    cleaned_data = clean_movies_dataset(data)
    return cleaned_data