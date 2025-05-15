
import pandas as pd
import matplotlib.pyplot as plt
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
import seaborn as sns

data = pd.read_json("Industrial_and_Scientific.json", lines=True)
# Print first 5 lines
print("head")
print(data.head())

#print info
print("info")
print(data.info())

#print describe
print("describe")
print(data.describe())

# Check for missing values in the dataset
print(data.isnull().sum())

# Drop rows where any of the specified columns have missing values
# Code adapted from W3Schools: Pandas DataFrame dropna() Method
# Retrieved from https://www.w3schools.com/python/pandas/ref_df_dropna.asp
data = data.dropna(subset=['reviewText', 'summary', 'vote','style' ])

# Fill missing values in reviewText with an empty string
# Code adapted from W3Schools: Pandas DataFrame fillna() Method
# Retrieved from https://www.w3schools.com/python/pandas/ref_df_fillna.asp
data['reviewerName'] = data['reviewerName'].fillna("")

# Drop column with too many missing values
# Code adapted from W3Schools: Pandas DataFrame drop() Method
# Retrieved from https://www.w3schools.com/python/pandas/ref_df_drop.asp
data = data.drop(columns=['image'])

#Recheck for missing values in the dataset
print(data.isnull().sum())

# Convert reviewTime to datetime
# Code adapted from W3Schools: Pandas - Cleaning Wrong Format
# Retrieved from https://www.w3schools.com/python/pandas/pandas_cleaning_wrong_format.asp
data['reviewTime'] = pd.to_datetime(data['reviewTime'])
print(data.head())

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to normalize text
# Code adapted from GeeksforGeeks: Removing Stop Words with NLTK in Python
# Retrieved from https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop-words
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Stemming
    # Code adapted from GeeksforGeeks: Introduction to Stemming
    # Retrieved from https://www.geeksforgeeks.org/introduction-to-stemming/?ref=header_outind
    stemmed = [stemmer.stem(word) for word in tokens]
    
    # Lemmatization
    # Code adapted from GeeksforGeeks: Python Lemmatization with NLTK
    # Retrieved from https://www.geeksforgeeks.org/python-lemmatization-with-nltk/?ref=header_outind#lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(lemmatized) 
#end of adapted code

 # Normalize text data
data['normalized_reviewText'] = data['reviewText'].apply(normalize_text)

print(data.head())

# Get unique ASINs 
unique_asins = data['asin'].unique()

# Display the first 5 unique ASINs
print("Unique ASINs:", unique_asins[:5])

# Select three unique product IDs
selected_asins =  [unique_asins[2], unique_asins[3], unique_asins[5], unique_asins[20]]
print("Selected ASINs:", selected_asins)


# Loop through the selected ASINs
for asin in selected_asins:
    # Filter data for the current ASIN
    product_data = data[data['asin'] == asin].copy()

    print(f"Processing ASIN: {asin}")
    print(f"Number of reviews: {product_data.shape[0]}")
    
    # Display summary statistics for ratings
    print(product_data['overall'].describe())
    

    # Plot rating distribution
    # Code adapted from W3Schools: Numpy Random Seaborn
    # Retrieved from https://www.w3schools.com/python/numpy/numpy_random_seaborn.asp
    sns.histplot(product_data['overall'], bins=5)
    plt.title(f"Rating Distribution for Product {asin}")
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.show()

    # Combine all reviews into one string
    all_reviews = ' '.join(product_data['normalized_reviewText'])

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

    # Code adapted from DataCamp: WordClouds in Python
    # Retrieved from https://app.datacamp.com/learn/tutorials/wordcloud-python
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis('off')
    plt.title(f"Word Cloud for Product {asin}")
    plt.show()

    
    # Group data by review time
    # Code adapted from Stack Overflow: Group by Month-Year in Pandas
    # Retrieved from https://stackoverflow.com/questions/66106578/how-can-i-make-a-group-by-month-year-converting-into-a-pandas-dataframe
    time_series = product_data.groupby(product_data['reviewTime'].dt.to_period('M')).size()

    # Plot the time series
    # Code adapted from GeeksforGeeks: Line Chart in Matplotlib
    # Retrieved from https://www.geeksforgeeks.org/line-chart-in-matplotlib-python/?ref=header_outind
    time_series.plot(kind='line', figsize=(10, 5))
    plt.title(f"Review Trends Over Time for Product {asin}")
    plt.xlabel("Time")
    plt.ylabel("Number of Reviews")
    plt.show()


    # Scatter plot of ratings and review length
    # Code adapted from Plotly: Line and Scatter Plots
    # Retrieved from https://plotly.com/python/line-and-scatter/
    product_data['reviewLength'] = product_data['normalized_reviewText'].apply(len)

    plt.scatter(product_data['reviewLength'], product_data['overall'], alpha=0.5)
    plt.title(f"Rating vs Review Length for Product {asin}")
    plt.xlabel("Review Length")
    plt.ylabel("Rating")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(product_data['reviewTime'], product_data['overall'], alpha=0.6, c='blue', edgecolors='w')
    plt.title("Product Ratings Over Time")
    plt.xlabel("Review Time")
    plt.ylabel("Rating")
    plt.grid(True)
    plt.show()
    #end of adapted code




