# Spotify Song Performance Analysis

## Tech Stack
### Language & Environment
- Python
- Google Colab

### Libraries Used
- `pandas`, `numpy` – Data manipulation
- `matplotlib.pyplot`, `plotnine` – Data visualization (ggplot-style)
- `re` – Regular expressions (text cleaning)
- `scikit-learn` –
  - **Models**:
    - Classification:  `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier`, `GaussianNB`, `CategoricalNB`
    - Regression: `LinearRegression`, `LassoCV`, `RidgeCV`, `GradientBoostingRegressor`
    - Clustering: `KMeans`, `DBSCAN`, `GaussianMixture`
  - **Preprocessing**: `StandardScaler`, `PolynomialFeatures`, `SplineTransformer`, `OneHotEncoder`, `LabelBinarizer`, `SimpleImputer`
  - **Pipelines**: `make_pipeline`, `Pipeline`, `make_column_transformer`
  - **Validation**: `train_test_split`, `KFold`, `LeaveOneOut`, `GridSearchCV`
  - **Evaluation**:
    - Regression: `mean_squared_error`, `mean_absolute_error`, `r2_score`
    - Classification: `accuracy_score`, `f1_score`, `recall_score`, `precision_score`, `roc_auc_score`
    - Visualization: `ConfusionMatrixDisplay`, `RocCurveDisplay`, `DecisionBoundaryDisplay`, `calibration_curve`
- `%matplotlib inline` – For plotting inside notebooks


## Introduction

Most Streamed Spotify Songs 2024 is a dataset obtained from Kaggle that lists variables from the top streamed songs on the music platform, Spotify. There were 29 columns and 4600 rows; however, nearly every part of the dataset contained missing values. Due to this, during the preprocessing portion of each model, we could not drop the missing values and instead replaced each missing value with a mean of its respective column. Further, the song titles contained special characters that needed to be dropped to create visualizations. The variables in the dataset are listed below:

#### Dataset Variable Descriptions

- `Track Name`: Name of the song  
- `Album Name`: Name of the album the song belongs to  
- `Artist`: Name of the artist(s) of the song  
- `Release Date`: Date when the song was released  
- `ISRC`: International Standard Recording Code for the song  

#### Popularity & Rankings
- `All Time Rank`: Ranking of the song based on its all-time popularity  
- `Track Score`: Score assigned to the track based on various factors  

#### Streaming Metrics
- `Spotify Streams`: Total number of streams on Spotify  
- `Spotify Playlist Count`: Number of Spotify playlists the song is included in  
- `Spotify Playlist Reach`: Reach of the song across Spotify playlists  
- `Spotify Popularity`: Popularity score of the song on Spotify  

#### YouTube Metrics
- `YouTube Views`: Total views of the song's official video on YouTube  
- `YouTube Likes`: Total likes on the song's official video on YouTube  
- `YouTube Playlist Reach`: Reach of the song across YouTube playlists  

#### TikTok Metrics
- `TikTok Posts`: Number of TikTok posts featuring the song  
- `TikTok Likes`: Total likes on TikTok posts featuring the song  
- `TikTok Views`: Total views on TikTok posts featuring the song  

#### Other Platforms
- `Apple Music Playlist Count`: Number of Apple Music playlists the song is included in  
- `Deezer Playlist Count`: Number of Deezer playlists the song is included in  
- `Deezer Playlist Reach`: Reach of the song across Deezer playlists  
- `Amazon Playlist Count`: Number of Amazon Music playlists the song is included in  
- `Pandora Streams`: Total number of streams on Pandora  
- `Pandora Track Stations`: Number of Pandora stations featuring the song  
- `Soundcloud Streams`: Total number of streams on Soundcloud  
- `TIDAL Popularity`: Popularity score of the song on TIDAL  

#### Radio & Other
- `AirPlay Spins`: Number of times the song has been played on radio stations  
- `SiriusXM Spins`: Number of times the song has been played on SiriusXM  
- `Shazam Counts`: Total number of times the song has been Shazamed  
- `Explicit Track`: Indicates whether the song contains explicit content  


Through conducting these analyses, we seek to gain insight on what impacts a song’s overall success. 



## Question 1: How does the song’s Explicit Track impact its performance on Spotify, YouTube, and TikTok?

### Methods
The goal of this analysis is to investigate how the presence of an "Explicit Track" label impacts the performance of a song on various platforms, including Spotify, YouTube, and TikTok. The analysis considers three key variables: Spotify Streams, YouTube Views, and TikTok Views, all of which are continuous variables, and Explicit Track, which is a binary variable (1 for explicit tracks and 0 for non-explicit tracks). By focusing on these variables, the analysis seeks to determine whether songs with explicit content perform differently across these platforms.

The first step in the analysis is to clean and prepare the dataset. Relevant columns are selected, ensuring that only the necessary data is retained. Any missing values are removed to maintain consistency in the dataset, and the indices are reset after dropping rows to keep the dataset well-organized. Non-numeric characters, such as commas, are also handled to ensure that the data is in the appropriate format for analysis. Once the data is cleaned, features (X) and target variables (y) are defined. The Explicit Track variable serves as the predictor (X), while Spotify Streams, YouTube Views, and TikTok Views are the target variables (y).

The next step involves splitting the data into training and testing sets, with an 80/20 split, using the train_test_split function. This split ensures that the models are properly validated, with 80% of the data used for training and 20% reserved for testing. Since the target variables are continuous, they are standardized using Z-score normalization to bring all target variables onto the same scale, making the models more effective. The Explicit Track variable, being binary, does not require any transformation.

For the modeling process, a Linear Regression model is fitted for each of the target variables: Spotify Streams, YouTube Views, and TikTok Views. The model uses Explicit Track as the predictor variable to assess how it influences each of these target variables. After fitting the models, various evaluation metrics are used to assess their performance. These metrics include Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values, Mean Absolute Error (MAE), which quantifies the average absolute difference, and the R² Score, which indicates how well the model explains the variance in the target variable. These metrics are calculated for both the training and testing sets for each platform.

To visualize the results, two types of graphs are created. The first is a bar chart that displays the R² scores for each Linear Regression model across all three target variables (Spotify Streams, YouTube Views, and TikTok Views) for both the training and testing sets. This chart allows for a clear comparison of the models' performance across platforms. The second visualization is a boxplot, which shows the relationship between the Explicit Track variable and each of the target variables. These plots help visualize whether explicit content has a noticeable impact on the performance of songs on each platform.

This analysis is effective in answering the research question by providing both quantitative and visual insights into how the "Explicit Track" feature affects the performance of songs across Spotify, YouTube, and TikTok. By evaluating the models using metrics like MSE, MAE, and R², the analysis shows how well the explicit track status predicts platform performance. The visualizations, including the bar chart and boxplots, make the findings easier to understand and offer a clear comparison of how explicit content influences each platform’s performance. This combined approach of statistical evaluation and visualization ensures a comprehensive understanding of the impact of explicit content on streaming and viewership.

### Results

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/81a491a4-a2c8-47fb-817b-e7806c54ee7e"><br>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/253ef0dd-be60-470d-80c9-94121c8d749d"><br>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f1c9a01b-eb9b-4d00-b9de-1c0d350b7edb"><br>
    </td>
  </tr>
</table>



For Spotify, the model's performance on both the training and test sets indicates a very weak correlation between the Explicit Track label and the number of streams. The R² values for both the training and testing sets are extremely low, close to 0, suggesting that the model does not explain the variance in Spotify Streams well. The negative R² on the test set indicates that the model's predictions perform worse than simply predicting the mean value of the target variable.

The model for YouTube shows a slight improvement over the Spotify model. The training R² is still low, at 0.0245, meaning the Explicit Track variable explains only a small portion of the variance in YouTube Views. The test R² is slightly positive but still very low, further indicating that the explicit nature of a track has minimal predictive power over the views on YouTube. While the model has marginal predictive value, it is not strong enough to make meaningful predictions based solely on the Explicit Track label.

The TikTok model shows even weaker results than Spotify and YouTube, with an extremely low training R² of 0.00025 and a negative test R², indicating that the explicit track label does not have a meaningful relationship with TikTok Views. Similar to Spotify, the model's performance is very poor, with the test set yielding results worse than predicting the mean.

The models' high MSE and MAE values across all platforms further confirm that predictions based solely on the explicit nature of a song are not reliable. While the data suggests some minimal correlation in specific cases, the explicit content alone is not a strong predictor of performance across these platforms.

<p align="center">
<img src="https://github.com/user-attachments/assets/62b2ae29-638b-449a-b6ea-15a74f62d60e" width="600" height="400"/>


Across all three platforms, the results suggest that the Explicit Track label has a very limited impact on the performance of songs. The R² values for both training and testing sets are low, and in many cases, negative, meaning that the models are not useful for predicting performance based on the explicit label. This indicates that factors other than whether a song is explicit likely play a much larger role in determining its performance on Spotify, YouTube, and TikTok.


<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9e8ee49d-609b-4a44-b063-0755020c1968">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/45d48a34-6fd9-4119-adb6-0cfe47062e56">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1892fe13-38b5-4d81-81a3-9cb26472b76c">
    </td>
  </tr>
</table>


The boxplots for Spotify Streams, YouTube Views, and TikTok Views by Explicit Track reveal minimal impact of the explicit content label on the performance of songs across these platforms. For Spotify, the distribution of streams for both explicit and non-explicit tracks shows a wide spread with some outliers, but the Explicit Track label does not appear to significantly affect the number of streams. Similarly, for YouTube, both explicit and non-explicit tracks exhibit tightly clustered views, indicating that the Explicit Track label has little influence on view counts, with only a few outliers in both categories. The TikTok Views boxplot is even more pronounced, as both explicit and non-explicit tracks are essentially squished to a line at zero, suggesting that explicit content has virtually no impact on TikTok engagement. Overall, these charts demonstrate that being an explicit track does not meaningfully alter the performance of songs on Spotify, YouTube, or TikTok, indicating that other factors likely play a larger role in driving views or streams on these platforms.

### Discussion/Reflection
The linear regression models offered insights into how the Explicit Track label might influence Spotify Streams, YouTube Views, and TikTok Views. However, the results indicated that explicit content has minimal impact on performance across these platforms, as evidenced by low R² values and compressed boxplots. This suggests that other factors, such as genre and artist popularity, likely play a more substantial role. As mentioned in the introduction, the preprocessing phase encountered issues due to widespread missing values throughout the dataset. Rather than dropping these rows, we replaced the missing values with the mean of their respective columns. This decision led to the expectation that the models would produce inaccurate or obscure results, which was not surprising.

While the models are useful for understanding the limited impact of the explicit label, they are constrained by the lack of key features such as genre, artist popularity, promotion strategies, and platform-specific factors like playlist reach and stream counts. Additionally, linear regression models are not well-suited to capturing complex, non-linear relationships, which likely contributed to the poor predictive performance. To improve the analysis, incorporating these relevant features and using advanced machine learning models, such as decision trees or random forests, would better capture non-linear relationships and interactions. Including time-series data and more platform-specific variables, like Shazam Counts and Pandora Streams, might further enhance the model’s predictive accuracy.



## Question 2: Group songs based on their performance across streaming platforms in terms of how many views/streams received on music platforms (Spotify, YouTube, TikTok, etc.). Can they be clustered appropriately this way? Use K-Means and evaluate their performance.

### Methods
The goal of this analysis is to group songs based on their performance across various streaming platforms, including Spotify, YouTube, TikTok, Pandora, and Soundcloud. By applying clustering techniques, specifically K-Means, we aim to determine if songs with similar streaming performance across these platforms can be effectively clustered together.

The first step was to select the relevant columns from the dataset. We then dropped any missing values to ensure consistency and reset the indices to maintain alignment after dropping rows with missing data. Since the dataset was unorganized, we created a new data frame containing the variables involved in the analysis.

To standardize the data and ensure fair comparison across features, we performed Z-score normalization on the continuous variables: Spotify Streams, YouTube Views, TikTok Views, Pandora Streams, and Soundcloud Streams. This step was essential for ensuring that each variable had the same scale before applying the clustering model. Once the data was scaled, we converted it back into a dataframe.

The next step involved visualizing the data before applying K-Means clustering. Scatterplots were created for two sets of variables: Spotify Streams vs. YouTube Views and Pandora Streams vs. Soundcloud Views. This was done to identify if any natural clusters appeared in the data, which would guide our decision on the appropriate number of clusters.

Using the scatterplot insights, we applied K-Means clustering to group the songs based on their performance. The number of clusters, n_clusters, was chosen based on the patterns observed in the scatterplots. We then fit the K-Means model to the scaled data and assigned a cluster label to each song. These labels were saved in a new column, clusters_km, in the dataset.

The results of the K-Means clustering were visualized with scatterplots, where each point was colored according to its assigned cluster label. This allowed us to visually assess how well the K-Means algorithm grouped the songs based on their performance across the streaming platforms.

### Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/32b305b1-e18e-4213-882f-85511b0ff26b" width="500" height="350">
  <img src="https://github.com/user-attachments/assets/4979d5c6-8bbd-41ed-8cd5-0a8a758d50b7" width="500" height="350">


Based on the scatterplot of Spotify Streams vs YouTube Views, where we applied K-Means with n_clusters = 2, the model does show some ability to group the data. However, the clustering results reveal some challenges. The two clusters appear to be mostly separated, but there is significant overlap, especially in the lower range of values. The larger cluster, which encompasses most of the data points, includes both songs with low and mid-range views/streams, making it less distinct than expected. Meanwhile, the smaller cluster contains only a few outliers, which are likely songs with exceptionally high performance on both platforms. Overall, while K-Means does provide some level of clustering, it doesn’t completely capture the underlying structure of the data, as the separation between clusters is not very clear. The significant overlap, particularly in the lower range of values, suggests that K-Means may not be the best method for this dataset without further tuning or additional features. Alternative clustering methods or a more nuanced choice of n_clusters could improve the results.



<p align="center">
  <img src="https://github.com/user-attachments/assets/d9a182f0-ed9a-47f3-8674-5c5875959fd8" width="500" height="350">
  <img src="https://github.com/user-attachments/assets/dcf797db-5718-4101-a246-5de7d0e89982" width="500" height="350">

  
Based on the scatterplot of Pandora Streams vs. SoundCloud Streams, where we applied K-Means with n_clusters = 3, the model reveals a mixed level of clustering success. While the model attempts to group the data into three distinct clusters, the results suggest significant overlap between some of these groups. Cluster 1, represented by green, effectively captures outliers or isolated groups with lower stream counts, indicating some level of accuracy in identifying distinct patterns. Cluster 0, represented in red, dominates the lower-left section of the graph, encompassing the majority of data points with low to moderate streams. However, this cluster shows considerable overlap with Cluster 2, represented in blue, particularly in the mid-range values of both SoundCloud and Pandora streams. The blue cluster appears to target medium to high-performing streams, but the overlap with the red cluster diminishes its distinctiveness. These results suggest that while K-Means provides a general grouping of the data, the clusters are not entirely separable. This may indicate that the underlying structure of the data does not conform well to three distinct clusters, or it may suggest that further feature engineering, preprocessing, or the selection of a different clustering approach could improve the model’s performance.

### Discussion/Reflection
Again, this analysis explored whether K-Means clustering could effectively group songs based on their performance across multiple music platforms, including Spotify, YouTube, Pandora, and Soundcloud. The results from applying K-Means with different values for n_clusters revealed that while the model could identify general patterns in the data, the clusters were not entirely distinct. For example, in the Spotify Streams vs YouTube Views analysis with n_clusters = 2, there was considerable overlap, especially in the lower range of values, making it difficult to fully separate the data into meaningful groups. Similarly, in the Pandora Streams vs SoundCloud Streams analysis with n_clusters = 3, the clustering showed better separation, but significant overlap remained, particularly between medium to high-performing streams. This suggests that while K-Means offers some insight into grouping songs, it may not be the best method for this dataset without further refinement.

The limitations of the K-Means model in this analysis stem from its assumptions of linear separability and fixed cluster shapes, which may not align well with the structure of this data. Additionally, the analysis was limited by the use of a narrow set of features—only streaming counts from a few platforms—without considering other potentially influential factors such as genre and artist popularity. To improve the clustering, experimenting with alternative clustering algorithms, such as DBSCAN, could help capture more complex, non-linear relationships. Including more features and performing additional preprocessing, such as handling outliers or adjusting for platform-specific factors, would provide a more accurate representation of song performance and potentially lead to more meaningful clusters.

