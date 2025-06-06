# Digital Magazine Customer Segmentation

## Tech Stack
### Language & Environment
- Python
- Google Colab

### Libraries Used
- `pandas`, `numpy` – Data manipulation
- `seaborn`, `plotnine`, `matplotlib.pyplot` – Data visualizations
- `scikit-learn` –
  - **Clustering**: `KMeans`, `DBSCAN`, `GaussianMixture`, `AgglomerativeClustering`
  - **Preprocessing**: `StandardScaler`, `NearestNeighbors`
  - **Evaluation**: `silhouette_score`, `f1_score`, `recall_score`, `accuracy_score`
  - **Dimensionality Reduction**: `PCA`
  - **Supervised Learning (optional)**: `LogisticRegression`, `train_test_split`
- `scikit-image` (`io`, `resize`) – For image loading and resizing
- `Pillow` (`PIL`) – Additional image handling
- `scipy.cluster.hierarchy` – Dendrograms and hierarchical clustering
- `%matplotlib inline` – For plotting inside notebooks


## Introduction

The objective of this analysis is to perform customer segmentation for a digital magazine. This involves clustering customers based on their behavioral attributes and article consumption patterns to gain insights into customer types and their needs. Behavioral clustering leverages demographic and engagement data, while article clustering analyzes reading patterns across various topics. Insights from these clusters can guide tailored marketing strategies and content recommendations, potentially improving user engagement and subscription rates. The behavioral dataset (behavioral.csv) contains information about the media company customers' behavior on the site. The variables in the customer data include:
- `id`: customer id
- `gender`: self-disclosed gender identity, `male`, `female`, `nonbinary` or `other`
- `age`: age in years
- `current_income`: self-reported current annual income in thousands
- `time_spent_browsing`: average number of minutes spent browsing website per month
- `prop_ad_clicks`: proportion of website ads that they click on (between `0` and `1`)
- `longest_read_time`: longest time spent consecutively on website in minutes
- `length_of_subscription`: number of days subscribed to the magazine
- `monthly_visits`: average number of visits to the site per month

The article dataset (topics.csv) contains information about the number of articles customers read in each topic in the past 3 months. The topics in the customer data include:
- `Stocks`
- `Productivity`
- `Fashion`
- `Celebrity`
- `Cryptocurrency`
- `Science`
- `Technology`
- `SelfHelp`
- `Fitness`
- `AI`


## Methods

To analyze customer behavior and cluster articles, we employed two different clustering models tailored to the type of data we were working with. Below, I outline the methods used for both behavioral clustering and article clustering, along with the steps taken to prepare the data and interpret the results.

### Behavioral Clustering Model

#### Pros and Cons
Clustering algorithms, each with unique strengths and limitations, play a crucial role in segmenting data effectively. Among the most popular are K-Means, DBSCAN, Hierarchical Clustering, and Gaussian Mixture Models (GMM). K-Means is widely recognized for its simplicity and computational speed. It performs exceptionally well with spherical, well-separated clusters in numerical datasets. However, its hard clustering approach, which forces each data point into a single cluster, may oversimplify complex relationships. Furthermore, K-Means is sensitive to outliers and requires specifying the number of clusters in advance, which can be a drawback when the optimal number of clusters is unknown.

DBSCAN offers unique advantages for identifying clusters of arbitrary shapes and handling noisy data effectively. It does not require predefining the number of clusters, making it flexible for exploratory analyses. However, its performance heavily depends on hyperparameter tuning, such as selecting an appropriate eps (neighborhood radius). DBSCAN also struggles with datasets containing clusters of varying densities, which can lead to inconsistent results. This makes it more suitable for dense, well-separated clusters with significant noise.

Hierarchical Clustering excels in providing a visual representation of relationships between data points through a dendrogram. This approach does not initially require specifying the number of clusters, allowing for exploratory flexibility. However, it is computationally intensive, especially for larger datasets, and sensitive to outliers that can distort the cluster hierarchy. Furthermore, selecting the cutoff threshold for defining clusters introduces subjectivity into the analysis. Hierarchical Clustering is better suited for smaller datasets where understanding relationships is a priority over scalability.

For this analysis, Gaussian Mixture Models (GMM) emerged as the best choice for behavioral clustering. Unlike K-Means, GMM uses soft clustering, assigning probabilities to each data point for belonging to different clusters. For pros, GMM can handle overlapping clusters, which is important when customer behaviors are not strictly distinct. It provides probabilistic cluster assignments, offering more flexibility than hard clustering methods. It also works well with continuous, standardized data. For cons, GMM can be sensitive to initialization and may converge to suboptimal solutions. It requires the number of clusters to be specified in advance, which can be subjective. GMM can struggle with high-dimensional data unless dimensionality reduction is applied. 

Here are the scatterplots created using all the variables/features (at least once):

<p align="center">
<img src="https://github.com/user-attachments/assets/20279447-4d01-4803-95bd-423601ace4ca" width="400" height="300"/> 
<img src="https://github.com/user-attachments/assets/78817ead-c344-4d89-922d-b7eb11ede939" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/1674efee-57ab-42e6-b49a-8bc96f4de89b" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/1886d017-f57d-4f01-b213-c1178d9ab27a" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/774ba68b-4c91-43c0-8f69-c0ef14fb16e3" width="400" height="300"/>



#### Chosen Model Details
For behavioral clustering, we used the Gaussian Mixture Model (GMM) to identify distinct groups of customers based on their activity and preferences. We will be focussing on time spent browsing vs the length of subscription. The GMM is a flexible clustering method that assumes data points come from a mixture of Gaussian distributions, allowing for “soft clustering.” This means each customer can belong to multiple clusters with varying probabilities, providing a nuanced view of customer behavior. 

The first step was to prepare the data by selecting key predictors of customer behavior. We used all the features in the dataset (except for ID), Since these features varied widely in scale (e.g., income vs. time spent browsing), we standardized the data using Z-scoring. Standardization ensures that all features have a mean of 0 and a standard deviation of 1, allowing the clustering algorithm to treat all features equally. After pre-processing, we fit the GMM model to the standardized data, specifying 4 clusters based on prior experimentation (second scatterplot above). 

To visualize the clusters, we created two types of plots: A scatterplot of time spent browsing versus length of subscription, colored by cluster, to highlight differences in customer engagement, and a Principal Component Analysis (PCA) plot, which reduced the data to two dimensions for easier visualization. PCA simplifies high-dimensional data, showing how clusters are distributed and whether they overlap.


### Article Clustering Model
For clustering articles, we used Hierarchical Agglomerative Clustering (HAC), a technique that groups data points into a hierarchy based on their similarity. This approach was ideal because it creates a tree-like structure, called a dendrogram, which visually shows how articles are grouped. HAC is particularly useful for datasets where the number of clusters is not known beforehand. We used all the features in the dataset (except for ID), These features represent the frequency with which each topic was associated with the articles. Since the data consisted of counts (e.g., how many times a topic was mentioned), standardization was unnecessary.

We applied cosine similarity as the distance metric (also called affinity) and average linkage (per instructions on the assignment). Linkage method: "Average," which determines cluster similarity by averaging pairwise distances between articles. Distance metric: "Cosine," which measures the similarity of articles based on the angle between their topic vectors. Initially, the model was fit with no pre-specified number of clusters, creating a dendrogram. The dendrogram helped us determine a meaningful cutoff point (0.5), which grouped the articles into five clusters. We then re-fit the model with this cluster count to assign each article to a specific group.


## Results
### Behavioral Clustering Model

<p align="center">
<img src="https://github.com/user-attachments/assets/fdfb9edc-dd3a-4461-a22e-45165729b47f" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/ecaa7c62-e218-44f2-8a16-b04fee303af6" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/efc1638b-c807-4673-93be-21dc968b6604">



The clusters in the GMM plot (Time Spent Browsing vs. Length of Subscription) are numbered starting from 0, 1, 2, 3. Cluster 0: Customers with low time spent browsing and high subscription length. Cluster 1: Customers with high time spent browsing and low subscription length. Cluster 2: Customers with high time spent browsing and high subscription length. Cluster 3: Customers with moderate browsing time and moderate subscription length.

The PCA plot visualizes the same clusters. The points are spread across the plot, clearly showing that the clusters have been separated well, and the GMM has done a good job of distinguishing them. The "other summary" table gives us the metrics for each cluster, and they correspond to the GMM clusters.

Understanding the behaviors of each cluster enables the company to recommend specific products or content based on user engagement. For instance: High engagement clusters can receive recommendations for premium features, upselling, or cross-selling opportunities. Low engagement clusters may benefit from simplified or more user-friendly features to boost their interaction with the platform. Additionally, analyzing customer behavior patterns can help the company identify which groups are at risk of churn (low engagement) and take proactive steps to retain them, such as sending reminders, offering discounts, or improving user experience for these customers.

### Article Clustering Model

<p align="center">

<img src="https://github.com/user-attachments/assets/6893f349-8aac-4e7c-919e-f366780abc1f">

<p align="center">

<img src="https://github.com/user-attachments/assets/e1f9fb5e-ba9e-467c-a5a1-343f9e7eaaba" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/cb87c1e2-36a3-487c-bd06-794cba3bf447" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/0cea093d-aef1-4d83-9533-0a33c985d2cb" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/5f9646a3-99ab-4f1c-882c-38689bf82cdc" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/d3eda2ff-079a-430c-926d-78c7acac864a" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/194fe14b-fc6f-40b2-b2e2-4b934bd1c196" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/0e03891e-5c81-406c-b07d-983f3bed8658" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/859fdf4a-e141-43c1-a1f4-80d2194e7e11" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/314a0b00-8d0c-48b5-8a47-7ca5a17eae26" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/bc6043df-fdac-48c7-8381-8acf0e78df0d" width="400" height="300"/>

<p align="center">
  
<img src="https://github.com/user-attachments/assets/e77c109a-388a-4cc8-b6a4-309cee4dabfd"/>



The dendrogram plots show how articles are grouped together. The clustering is done based on content similarity, and the branches indicate these groupings. Cluster 0: Likely related to fashion, science, and fitness content. Appears on the leftmost part of the dendrogram with articles that fit this thematic grouping. Cluster 1: Articles around stocks, productivity, and self-help are visible in the middle of the dendrogram. Cluster 2: Cluster centered around science, technology, and AI content, found toward the right side of the dendrogram. Cluster 3: Focuses on some stocks, some celebrity, and mostly cryptocurrency articles, which are grouped closer to the end of the dendrogram on the right. Cluster 4: Articles focusing on fashion and celebrity, situated in the far-right part of the dendrogram.

The article clustering results can significantly enhance the company’s ability to personalize user experiences, optimize content strategies, and improve ad targeting. By understanding which topics resonate with specific user segments, the company can tailor content recommendations accordingly. For example, Cluster 0, which includes fashion, science, and fitness topics, can be targeted to users interested in health and lifestyle content, while Cluster 2, focused on technology and AI, can be aimed at users with an interest in innovation. This allows the company to deliver relevant content to users, increasing engagement and user satisfaction. Furthermore, by identifying popular and underrepresented topics, the company can better prioritize content creation and fill gaps in their offering.


## Discussion/Reflection

Performing these clustering analyses helped me gain insights into how customer behaviors and article content can be grouped based on similarities, providing valuable information for more targeted strategies, such as personalized content delivery or marketing campaigns. I learned that techniques like GMM and hierarchical clustering are effective in uncovering distinct patterns within data, which can drive decision-making. If I were to perform this analysis again, I would explore other clustering algorithms or fine-tune the current models further to improve cluster separation. Additionally, incorporating more features, such as user interaction history or sentiment analysis of articles, could enhance the segmentation and lead to more personalized recommendations.



