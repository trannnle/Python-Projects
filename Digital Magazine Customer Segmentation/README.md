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
<img src="https://github.com/user-attachments/assets/14ae7407-481b-48c2-8bd1-ddc9537f4bab" width="400" height="300"/> 
<img src="https://github.com/user-attachments/assets/3d49df6c-2eac-459a-81f8-498ba6c76362" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/c6b9a8ce-6744-492a-a919-5473696cc3d9" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/e10a0cc7-21db-40cc-80fe-dc29d67d06eb" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/58d44eeb-e1cf-47fd-bffe-34db5be654e3" width="400" height="300"/>



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
<img src="https://github.com/user-attachments/assets/c1100020-3a69-4f09-b29c-6aefc36fbec2" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/88e7fbde-a0e7-4e45-8d8e-9b0bc4369eb7" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/73d025f6-9b3b-47dc-a129-6b1cdbe1f0e2">



The clusters in the GMM plot (Time Spent Browsing vs. Length of Subscription) are numbered starting from 0, 1, 2, 3. Cluster 0: Customers with low time spent browsing and high subscription length. Cluster 1: Customers with high time spent browsing and low subscription length. Cluster 2: Customers with high time spent browsing and high subscription length. Cluster 3: Customers with moderate browsing time and moderate subscription length.

The PCA plot visualizes the same clusters. The points are spread across the plot, clearly showing that the clusters have been separated well, and the GMM has done a good job of distinguishing them. The "other summary" table gives us the metrics for each cluster, and they correspond to the GMM clusters.

Understanding the behaviors of each cluster enables the company to recommend specific products or content based on user engagement. For instance: High engagement clusters can receive recommendations for premium features, upselling, or cross-selling opportunities. Low engagement clusters may benefit from simplified or more user-friendly features to boost their interaction with the platform. Additionally, analyzing customer behavior patterns can help the company identify which groups are at risk of churn (low engagement) and take proactive steps to retain them, such as sending reminders, offering discounts, or improving user experience for these customers.

### Article Clustering Model

<p align="center">

<img src="https://github.com/user-attachments/assets/9d9421d3-aa22-4d48-80b9-f46c5429faa0">

<p align="center">

<img src="https://github.com/user-attachments/assets/55b973eb-3666-47d1-b5a7-f9b438609468" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/0bbbf379-cf70-4178-aaf1-383209a3e8ac" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/a89787e7-d14c-4e08-81db-f9a07ab7e58d" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/a6ccc50f-34ac-44c6-b994-68019bed1c1b" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/4443ae23-f301-4446-858a-a22a80543de5" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/228729c7-faaa-4b35-8f45-34317155b39a" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/ffc5febf-85c4-4542-aaad-82cc960782c1" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/d6f0a77e-754b-4096-87ef-d6dc3175c94b" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/59100009-3bb0-4396-8761-b4cf785bccdb" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/2aae12c8-d44c-4a65-925a-55f18680bbaa" width="400" height="300"/>

<p align="center">
  
<img src="https://github.com/user-attachments/assets/db68cabe-9689-413d-8738-c9aa04cd0e8c"/>



The dendrogram plots show how articles are grouped together. The clustering is done based on content similarity, and the branches indicate these groupings. Cluster 0: Likely related to fashion, science, and fitness content. Appears on the leftmost part of the dendrogram with articles that fit this thematic grouping. Cluster 1: Articles around stocks, productivity, and self-help are visible in the middle of the dendrogram. Cluster 2: Cluster centered around science, technology, and AI content, found toward the right side of the dendrogram. Cluster 3: Focuses on some stocks, some celebrity, and mostly cryptocurrency articles, which are grouped closer to the end of the dendrogram on the right. Cluster 4: Articles focusing on fashion and celebrity, situated in the far-right part of the dendrogram.

The article clustering results can significantly enhance the company’s ability to personalize user experiences, optimize content strategies, and improve ad targeting. By understanding which topics resonate with specific user segments, the company can tailor content recommendations accordingly. For example, Cluster 0, which includes fashion, science, and fitness topics, can be targeted to users interested in health and lifestyle content, while Cluster 2, focused on technology and AI, can be aimed at users with an interest in innovation. This allows the company to deliver relevant content to users, increasing engagement and user satisfaction. Furthermore, by identifying popular and underrepresented topics, the company can better prioritize content creation and fill gaps in their offering.


## Discussion/Reflection

Performing these clustering analyses helped me gain insights into how customer behaviors and article content can be grouped based on similarities, providing valuable information for more targeted strategies, such as personalized content delivery or marketing campaigns. I learned that techniques like GMM and hierarchical clustering are effective in uncovering distinct patterns within data, which can drive decision-making. If I were to perform this analysis again, I would explore other clustering algorithms or fine-tune the current models further to improve cluster separation. Additionally, incorporating more features, such as user interaction history or sentiment analysis of articles, could enhance the segmentation and lead to more personalized recommendations.



