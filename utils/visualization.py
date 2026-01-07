import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bar visualization of rating counts distribution
def plot_rating_counts(df: pd.DataFrame) -> None:
    """
    Visualizes the distribution of ratings in the dataset.
    This will highlight the frequency of each rating value.
    :param df:
    :return: None (displays a bar plot)
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rating', data=df, palette='viridis')
        plt.title('Rating Counts Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting rating counts: {e}")

# Bar visualization of genre counts distribution
def plot_genre_counts(df: pd.DataFrame) -> None:
    """
    Visualizes the distribution of movie genres in the dataset.
    This will highlight the popularity of different genres.
    :param df:
    :return: None (displays a bar plot)
    """
    try:
        # Data already processed to have list of genres in the dataframe instead of a string with '|' delimiter
        all_genres = df['genres'].explode()
        genre_counts = all_genres.value_counts()
        plt.figure(figsize=(12, 8))
        sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='magma')
        plt.title('Genre Counts Distribution')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting genre counts: {e}")

# Histogram visualization of user rating count distribution
def plot_user_activity(df: pd.DataFrame) -> None:
    """
    Visualizes the distribution of the number of ratings per user.
    This will highlight the activity level of users in the dataset.
    :param df:
    :return: None (displays a histogram)
    """
    try:
        user_rating_counts = df['userId'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(user_rating_counts, bins=30, kde=False, color='skyblue')
        plt.title('User Rating Count Distribution')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting user activity: {e}")

# Scatter plot of movie popularity vs average rating
def plot_popularity_vs_average(df: pd.DataFrame) -> None:
    """
    Visualizes the relationship between movie popularity (number of ratings) and average rating.
    This will help identify trends between how often a movie is rated and its average score.
    :param df:
    :return: None (displays a scatter plot)
    """
    try:
        movie_stats = df.groupby('movieId').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['average_rating', 'rating_count']
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='rating_count', y='average_rating', data=movie_stats, alpha=0.6)
        plt.title('Movie Popularity vs Average Rating')
        plt.xlabel('Number of Ratings (Popularity)')
        plt.ylabel('Average Rating')
        plt.xscale('log')
        plt.ylim(0, 5)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting popularity vs average rating: {e}")

# Bar chart comparing model performance
def plot_model_comparison(results: dict) -> None:
    """
    Compares the performance of different models using a bar chart.
    :param results: Dictionary with model names as keys and metric values as values.
    :return: None (displays a bar plot)
    """
    try:
        metrics_df = pd.DataFrame(results).T # Convert dict to DataFrame for plotting
        metrics_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison (RMSE & MAE)')
        plt.ylabel('Error Value (Lower is Better)')
        plt.xticks(rotation=0)
        plt.legend(title='Metrics')
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting model comparison: {e}")