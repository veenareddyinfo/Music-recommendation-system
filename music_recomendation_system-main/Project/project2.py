import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from flask import Flask, request, render_template
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "Enter your client id here"
client_secret = "Enter your client secret here"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

df = pd.read_csv("C:\Python\Project\dataset.csv")

numeric_features = [
    "popularity", "duration_ms", "danceability", "energy", "key", "loudness", 
    "mode", "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo", "time_signature"
]

df_sample = df[["track_id", "track_name", "album_name", "artists"] + numeric_features].dropna().reset_index(drop=True)

app = Flask(__name__)

def perform_clustering(num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_sample['cluster'] = kmeans.fit_predict(df_sample[numeric_features])
    return kmeans

kmeans_model = perform_clustering(num_clusters=5)

def get_album_image_and_url(song_name, artist_name):
    result = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
    if result['tracks']['items']:
        track = result['tracks']['items'][0]
        album_image_url = track['album']['images'][0]['url']  
        spotify_url = track['external_urls']['spotify']       
        return album_image_url, spotify_url
    return None, None

def recommender(track_name, num_recommendations=10):
    try:
        idx = df_sample[df_sample["track_name"].str.lower() == track_name.lower()].index[0]
    except IndexError:
        return []  
    
    song_vector = df_sample.loc[idx, numeric_features].values.reshape(1, -1)
    song_cluster = df_sample.loc[idx, 'cluster']
    cluster_songs = df_sample[df_sample['cluster'] == song_cluster]
    all_vectors = cluster_songs[numeric_features].values
    similarity_scores = cosine_similarity(song_vector, all_vectors).flatten()
    similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]
    recommended_songs = cluster_songs.iloc[similar_indices]
    
    recommendations = []
    for _, row in recommended_songs.iterrows():
        song_name = row["track_name"]
        artist_name = row["artists"]
        album_image_url, spotify_url = get_album_image_and_url(song_name, artist_name)
        duration_ms = row["duration_ms"]
        duration_formatted = f"{int(duration_ms // 60000)}:{int((duration_ms % 60000) // 1000):02}"  # Convert ms to mm:ss
        
        recommendations.append({
            "name": song_name,
            "image": album_image_url,
            "duration": duration_formatted,
            "spotify_url": spotify_url
        })
    
    return list({song['name']: song for song in recommendations}.values())

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        user_input = request.form["song_name"].strip()
        recommendations = recommender(user_input)
    
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
