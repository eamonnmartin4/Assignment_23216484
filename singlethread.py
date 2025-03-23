import pandas as pd
from pandas import merge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

#start_time_single_pre = time.time()
#Update these paths, download of files available from my Repo's ReadMe file
file_paths = {
    "spotify.csv": "/Users/eamonn/Desktop/BigData/Assignment/charts.csv",
    "spotify_audio.csv": "/Users/eamonn/Desktop/BigData/Assignment/Spotify_Dataset_V3.csv",
    "hot100.csv": "/Users/eamonn/Desktop/BigData/Assignment/Hot Stuff.csv",
    "hot100_audio.csv": "/Users/eamonn/Desktop/BigData/Assignment/Hot 100 Audio Features.csv"
}

#Make dataframes out of our 4 csv's
spotify_charts = pd.read_csv(file_paths["spotify.csv"])
spotify_audio = pd.read_csv(file_paths["spotify_audio.csv"], delimiter=';')
hot100 = pd.read_csv(file_paths["hot100.csv"])
hot100_audio = pd.read_csv(file_paths["hot100_audio.csv"])

print("Missing values in Spotify Charts dataset:\n", spotify_charts.isnull().sum())
print("Missing values in Spotify Audio Features dataset:\n", spotify_audio.isnull().sum())
print("Missing values in Hot 100 dataset:\n", hot100.isnull().sum())
print("Missing values in Hot 100 Audio Features dataset:\n", hot100_audio.isnull().sum())

'''
print("Memory usage of datasets (in MB) After Loading:")
print("Spotify Charts:", spotify_charts.memory_usage(deep=True).sum() / 1e6)
print("Spotify Audio:", spotify_audio.memory_usage(deep=True).sum() / 1e6)
print("Hot100:", hot100.memory_usage(deep=True).sum() / 1e6)
print("Hot100 Audio:", hot100_audio.memory_usage(deep=True).sum() / 1e6)
'''

#Remove rows where 'title' or 'artist' is missing, as they are useless for merging without these
spotify_charts.dropna(subset=['title', 'artist'], inplace=True)
spotify_charts.reset_index(drop=True, inplace=True)
#end_time_single_pre = time.time()

#execution_time_single_pre = end_time_single_pre - start_time_single_pre  # Calculate time taken
#print(f"Execution time: {execution_time_single_pre} seconds")

#List of numeric audio feature columns to fill
audio_feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

#Filtering
hot100_audio = hot100_audio.dropna(subset=audio_feature_cols)
hot100_audio.dropna(subset=['spotify_genre'], inplace=True)
hot100_audio = hot100_audio[~hot100_audio['spotify_genre'].str.contains("\[]", na=False)]

'''
print("Hot100:", hot100.memory_usage(deep=True).sum() / 1e6)
print("Hot100 Audio:", hot100_audio.memory_usage(deep=True).sum() / 1e6)
'''
#start_time_single_merge1 = time.time()
hot100_merged = merge(hot100, hot100_audio, on="SongID", how="left", copy=False)
print("Hot100 Datasets merged successfully! ✅")
#end_time_single_merge1 = time.time()
#execution_time_single_merge1 = end_time_single_merge1 - start_time_single_merge1  # Calculate time taken
#print(f"Execution time: {execution_time_single_merge1} seconds")

#print("Hot100 Merged:", hot100_merged.memory_usage(deep=True).sum() / 1e6)

#start_time_single_pre2 = time.time()
#Ensure "Title" and "Artists" are strings
spotify_audio["Title"] = spotify_audio["Title"].astype(str).str.lower().str.strip()
spotify_audio["Artists"] = spotify_audio["Artists"].astype(str).str.lower().str.strip()

#Convert numeric_cols to a list
numeric_cols = list(spotify_audio.select_dtypes(include=["number"]).columns)

#Aggregate only numeric columns, keep first occurrence of other columns
spotify_audio = spotify_audio.groupby(["Title", "Artists"], as_index=False).agg({
    **{col: "mean" for col in numeric_cols},  #Take mean for numeric columns
    **{col: "first" for col in [col for col in spotify_audio.columns if col not in numeric_cols + ["Title", "Artists"]]}  #Keep first for others
})

#end_time_single_pre2 = time.time()
#execution_time_single_pre2 = end_time_single_pre2 - start_time_single_pre2  # Calculate time taken
#print(f"Execution time: {execution_time_single_pre2} seconds")

#print("Spotify Charts:", spotify_charts.memory_usage(deep=True).sum() / 1e6)
#print("Spotify Audio:", spotify_audio.memory_usage(deep=True).sum() / 1e6)

#start_time_single_merge2 = time.time()
#create dictionary
spotify_audio_dict = spotify_audio.set_index(["Title", "Artists"]).to_dict(orient="index")

chunk_size = 50000
chunks = []
def process_chunk(chunk):
    #normalize song & artist names
    chunk["title"] = chunk["title"].str.lower().str.strip()
    chunk["artist"] = chunk["artist"].str.lower().str.strip()

    #add audio features using dictionary lookup
    features = [spotify_audio_dict.get((row["title"], row["artist"]), {}) for _, row in chunk.iterrows()]

    #convert list of dictionaries into DataFrame
    features_df = pd.DataFrame(features)

    #merge chunk with extracted features
    return pd.concat([chunk.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

spotify_chunks = pd.read_csv(file_paths["spotify.csv"], chunksize=chunk_size)

spotify_processed_chunks = []
for chunk in spotify_chunks:
    #normalize title & artist names in charts
    chunk["title"] = chunk["title"].str.lower().str.strip()
    chunk["artist"] = chunk["artist"].str.lower().str.strip()

    #filter songs that exist in the Spotify Audio dataset
    chunk = chunk[chunk.set_index(["title", "artist"]).index.isin(spotify_audio_dict.keys())]

    #add audio features
    processed_chunk = process_chunk(chunk)

    #append to final list
    spotify_processed_chunks.append(processed_chunk)

#merge all processed chunks
spotify_merged = pd.concat(spotify_processed_chunks, ignore_index=True)

#remove any remaining songs without audio feature values
spotify_merged.dropna(subset=numeric_cols, inplace=True)
spotify_merged.drop_duplicates(subset =["title"], keep="first", inplace=True)

print("Spotify datasets merged successfully with only matched songs! ✅")
#end_time_single_merge2 = time.time()

spotify_merged.drop(columns=['Loudness', 'url', 'Rank', 'Points (Ind for each Artist/Nat)', 'Date', '# of Artist', 'Artist (Ind.)', '# of Nationality', 'Song URL'], inplace=True)
#execution_time_single_merge2 = end_time_single_merge2 - start_time_single_merge2  # Calculate time taken
#print(f"Execution time: {execution_time_single_merge2} seconds")

#print("Spotify Merged:", spotify_merged.memory_usage(deep=True).sum() / 1e6)

start_time_single_analysis = time.time()
audio_features_spotify = ['Danceability', 'Energy', 'Speechiness', 'Acousticness',
       'Instrumentalness', 'Valence']

X = spotify_merged[audio_features_spotify].fillna(0)
y = spotify_merged["Points (Total)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

importance = model.coef_
feature_importance_spotify = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

#print("Spotify Audio Feature Importance: ", feature_importance_spotify)

audio_feature_hot = ["danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "valence"]

X = hot100_merged[audio_feature_hot].fillna(0)
y = hot100_merged["Peak Position"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

importance = model.coef_
feature_importance_hot100 = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

print("Billboard100 Audio Feature Importance: ", feature_importance_hot100)
#end_time_single_analysis = time.time()

#execution_time_single_analysis = end_time_single_analysis - start_time_single_analysis  # Calculate time taken
#print(f"Execution time: {execution_time_single_analysis} seconds")
#