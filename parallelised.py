import pandas as pd
from pandas import merge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pyspark import SparkContext
from pyspark import SparkConf
import time


#Set up spark
conf = SparkConf() \
    .setAppName("BigData_Assignment") \
    .set("spark.driver.memory", "8g") \
    .set("spark.executor.memory", "8g") \
    .set("spark.driver.maxResultSize", "4g")

sc = SparkContext("local[*]", "BigData_Assignment", conf=conf)
sc.setLogLevel("ERROR")

#Update these paths, download of files available from my Repo's ReadMe file
file_paths = {
    "spotify.csv": "/Users/eamonn/Desktop/BigData/Assignment/charts.csv",
    "spotify_audio.csv": "/Users/eamonn/Desktop/BigData/Assignment/Spotify_Dataset_V3.csv",
    "hot100.csv": "/Users/eamonn/Desktop/BigData/Assignment/Hot Stuff.csv",
    "hot100_audio.csv": "/Users/eamonn/Desktop/BigData/Assignment/Hot 100 Audio Features.csv"
}

hot100 = pd.read_csv(file_paths["hot100.csv"])
hot100_audio = pd.read_csv(file_paths["hot100_audio.csv"])

#List of numeric audio feature columns to fill
audio_feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

hot100_audio = hot100_audio.dropna(subset=audio_feature_cols)
hot100_audio.dropna(subset=['spotify_genre'], inplace=True)
hot100_audio = hot100_audio[~hot100_audio['spotify_genre'].str.contains("\[]", na=False)]

hot100_merged = merge(hot100, hot100_audio, on="SongID", how="left", copy=False)
print("Hot100 Datasets merged successfully! ✅")

start_time_mult = time.time()

spotify_charts_rdd = sc.textFile(file_paths["spotify.csv"])
spotify_audio_rdd = sc.textFile(file_paths["spotify_audio.csv"])

spotify_charts_rdd = spotify_charts_rdd.map(lambda line: line.split(','))
spotify_audio_rdd = spotify_audio_rdd.map(lambda line: line.split(';'))

#filter out rows where 'title' or 'artist' are missing
spotify_charts_rdd = spotify_charts_rdd.filter(lambda row: len(row) > 2 and row[1].strip() != "" and row[2].strip() != "")
spotify_audio_rdd = spotify_audio_rdd.filter(lambda row: len(row) > 2 and row[0].strip() != "" and row[1].strip() != "")

def normalize(row, title_idx, artist_idx):
    row[title_idx] = row[title_idx].strip().lower()  #normalize title

    artist = row[artist_idx].strip().lower()
    artist_list = artist.split(',')
    row[artist_idx] = artist_list[0].strip()
    return row

spotify_charts_rdd = spotify_charts_rdd.map(lambda row: normalize(row, 0, 3))  #title at index 1, artist at index 2
spotify_audio_rdd = spotify_audio_rdd.map(lambda row: normalize(row, 1, 2))  #title at index 0, artist at index 1

audio_feature_indices = [4, 5, 7, 8, 9, 10, 16]

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def valid_row(row, indices):
    return all(is_numeric(row[i]) for i in indices)

#remove invalid rows
spotify_audio_rdd = spotify_audio_rdd.filter(lambda row: valid_row(row, audio_feature_indices))

#convert each row to (key, values) -> key = (title, artist), values = list of numeric features
spotify_audio_rdd = spotify_audio_rdd.map(lambda row: (
    (row[1], row[2]), [float(row[i]) for i in audio_feature_indices]
))

#compute the mean for each key using (sum, count)
spotify_audio_rdd = spotify_audio_rdd.mapValues(lambda values: (values, 1))  # (values, count)

#use first occurrence
spotify_audio_rdd = spotify_audio_rdd.reduceByKey(lambda a, b: a)

#compute mean by dividing sum by count
spotify_audio_rdd = spotify_audio_rdd.mapValues(
    lambda x: [val / x[1] for val in x[0]]
)
#convert spotify_charts_rdd into key-value pairs
spotify_charts_rdd = spotify_charts_rdd.map(lambda row: ((row[0], row[3]), row))  # (title, artist) as key

#join on (title, artist)
spotify_merged_rdd = spotify_charts_rdd.join(spotify_audio_rdd)

#flatten the merged dataset
spotify_merged_rdd = spotify_merged_rdd.map(lambda x: x[1][0] + x[1][1])

spotify_merged_list = spotify_merged_rdd.collect()
spotify_merged_df = pd.DataFrame(spotify_merged_list)

print("Spotify datasets merged successfully using RDDs ✅")
end_time_mult = time.time()
execution_time_mult = end_time_mult - start_time_mult
print(f"Execution time: {execution_time_mult} seconds")

#add names to columns
columns = [
    'title', 'rank', 'date', 'artist', 'url', 'region', 'chart',
    'trend', 'streams', 'danceability', 'energy',
    'speechiness', 'acousticness', 'instrumentalness', 'valence', 'Points (Total)', 'x'
]
spotify_merged_df.columns = columns



spotify_merged_df.drop_duplicates(subset =["title"], keep="first", inplace=True)

audio_feature = ["danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "valence"]

X = spotify_merged_df[audio_feature].fillna(0)
y = spotify_merged_df["Points (Total)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

importance = model.coef_
feature_importance_spotify = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

print("Spotify Audio Feature Importance: ", feature_importance_spotify)

X = hot100_merged[audio_feature].fillna(0)
y = hot100_merged["Peak Position"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

importance = model.coef_
feature_importance_hot100 = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

print("Billboard100 Audio Feature Importance: ", feature_importance_hot100)
#
