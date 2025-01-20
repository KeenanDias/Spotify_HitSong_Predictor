---

# Spotify_HitSong_Predictor

## Introduction
In this project, I used a dataset called Spotify Most Streamed Songs. Using this dataset, I will create a model to answer my research question by analyzing many attributes of these songs. This model should help develop an idea of how to create viral songs by using the attributes of the songs in the dataset.

## Data Description
The dataset being used for this assignment is called Spotify Most Streamed Songs. This dataset holds many song attributes:

- **Back Track Information**: `track_name`, `artist(s)_name`, `artist_count`, `released_year`, `released_month`, `released_day`.
- **Streaming Metrics**: `in_spotify_playlists`, `in_spotify_charts`, `streams` (total number of streams on Spotify), `in_apple_playlists`, `in_apple_charts`, `in_deezer_playlists`, `in_deezer_charts`, `in_shazam_charts`.
- **Musical Attributes**: 
  - `bpm`: Beats per minute, representing the tempo of the song.
  - `key`: Key of the song.
  - `mode`: Indicates whether the song is in a major or minor mode.
  - `danceability_%`: Suitability of the song for dancing.
  - `valence_%`: Positivity of the song’s musical content.
  - `energy_%`: Perceived energy level of the song.
  - `acousticness_%`: Acoustic sound presence in the song.
  - `instrumentalness_%`: Proportion of instrumental content in the track.
  - `liveness_%`: Presence of live performance elements.
  - `speechiness_%`: Amount of spoken words in the song.

## Research Question
Will my song reach a billion or more streams? This question will help me analyze the song attributes in order to see why a song succeeds.

## Code Explanation
```python
Spotify_data = pd.read_csv('Spotify Most Streamed Songs.csv')
Spotify_data.head()
```
This code lets me load and view the first 5 rows of the dataset.

```python
Spotify_data.drop(['artist(s)_name','track_name'], axis=1, inplace=True)
Spotify_data.dropna(inplace=True)
```
Drops any non-relevant columns and drops any rows with missing values.

```python
Spotify_data['streams'] = pd.to_numeric(Spotify_data['streams'], errors='coerce')
```
This code makes `streams` a numeric value and the `errors='coerce'` is for values where there are non-numeric values and changes them to NaN.

```python
Spotify_data['Over_1_Billion'] = Spotify_data['streams'] > 1000000000
```
This code creates a variable to identify if a song has more than a billion streams.

```python
Song_Characteristics = ['danceability_%', 'energy_%', 'liveness_%', 'speechiness_%', 'in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
X = Spotify_data[Song_Characteristics]
y = Spotify_data['Over_1_Billion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=87)
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)
```
I used a logistic regression model and trained it, splitting `X` and `y` where `X` is the Song_Characteristics as Musical attributes.

```python
my_test = {
 'danceability_%': [76],
 'valence_%': [50],
 'energy_%': [60],
 'acousticness_%': [10],
 'instrumentalness_%': [5],
 'liveness_%': [10],
 'speechiness_%': [5],
 'bpm': [90]
}
df = pd.DataFrame(my_test)
print(logmodel.predict(df))
```
`my_test` holds the musical attributes where we can input attributes of our song to predict if our song will go viral with more than a billion streams on Spotify.

## Graphs

### Graph 1:
![image](https://github.com/user-attachments/assets/e048967a-4c04-4811-8c8c-a14221870992)

**Code:**
```python
sns.histplot(Spotify_data['streams'])
plt.title('Distribution of Streams')
plt.xlabel('Streams')
plt.ylabel('Count')
```
**Description:** This is a histogram that shows the distribution of stream counts. This shows that there are 175 songs that have 100 thousand streams and there are fewer songs that have 2.7 million streams.

### Graph 2:
![image](https://github.com/user-attachments/assets/fdba514d-187a-4134-8f3c-8dbf39940b2c)

**Code:**
```python
sns.scatterplot(x='danceability_%', y='streams', data=Spotify_data)
plt.title('Danceability vs. Streams')
plt.xlabel('Danceability (%)')
plt.ylabel('Streams')
```
**Description:** The scatterplot shows us the relationship between a song's danceability and the number of times the song has been streamed on Spotify. Each dot is a song and the more right on the x-axis the more danceable it is, the higher the dot is the more streams it has.

### Graph 3:
![image](https://github.com/user-attachments/assets/a1ec5fec-74e9-48dc-829b-56f6b59f6fd5)

**Code:**
```python
sns.scatterplot(x='energy_%', y='streams', data=Spotify_data)
plt.title('Energy vs. Streams')
plt.xlabel('Energy (%)')
plt.ylabel('Streams')
```
**Description:** This scatter plot shows the relationship between the energy the song gives in percentage vs how many times the song has been streamed on Spotify. The higher the dot the more streams it has, the dot that is more to the right the higher the energy level it has.

### Graph 4:
![image](https://github.com/user-attachments/assets/a518bd63-b8e7-4db2-ae7b-1a594ab5b74b)

**Code:**
```python
sns.scatterplot(x='speechiness_%', y='streams', data=Spotify_data)
plt.title('Speechiness vs. Streams')
plt.xlabel('Speechiness (%)')
plt.ylabel('Streams')
```
**Description:** This scatter plot shows the relationship between speechiness vs streams. The more words spoken in the song the more the dot is going to be on the right and the higher the streams the song will be higher up vertically on the graph.

### Graph 5:
![image](https://github.com/user-attachments/assets/07588bda-27af-434c-b5f6-88dbb6944832)

**Code:**
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='artist_count', y='streams', data=Spotify_data)
plt.title('Number of Artist(s) vs. Streams')
plt.xlabel('Number of Artist(s)')
plt.ylabel('Streams')
```
**Description:** This is a scatter plot that shows the number of artists vs streams. More viral songs are guaranteed to have fewer artists on their songs.

## Conclusion
In conclusion, certain musical and streaming attributes can help predict whether a song can go viral and have a billion or more streams. However, my model is not perfect and has a lot of room to improve. It creates a starting point for users who want to create a viral song using this model.

## Use Of New Material
- python - Change column type in pandas - Stack Overflow 
  - Used to make ‘streams’ column numeric to search for 1 billion streams
  - Error handling using error=coerce for values that are non-numeric to be changed to NaN.
- Spotify Most Streamed Songs (kaggle.com) 

---
