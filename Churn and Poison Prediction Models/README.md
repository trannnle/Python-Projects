# Churn & Poison Prediction Models

## Tech Stack
### Language & Environment
- Python
- Google Colab

### Libraries Used
- `pandas`, `numpy` – Data manipulation
- `plotnine`, `matplotlib.pyplot` – Data visualizations
- `scikit-learn` –
  - **Models**: `LogisticRegression`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `GaussianNB`, `CategoricalNB`, `DecisionTreeClassifier`
  - **Preprocessing**: `StandardScaler`, `OneHotEncoder`, `LabelBinarizer`
  - **Validation**: `train_test_split`, `KFold`, `LeaveOneOut`, `GridSearchCV`
  - **Evaluation**: `accuracy_score`, `f1_score`, `recall_score`, `precision_score`, `roc_auc_score`, `confusion_matrix`
  - **Visualization**: `ConfusionMatrixDisplay`, `RocCurveDisplay`, `DecisionBoundaryDisplay`, `calibration_curve`
- `%matplotlib inline` – For plotting inside notebooks

## Introduction
This report focuses on using predictive models to solve two distinct problems. 

The first problem involves predicting customer churn for a streaming service using a business churn dataset (streaming.csv). Churn refers to when customers stop using a service, which is critical for streaming companies because it directly impacts their revenue and growth. By using data about customer characteristics—like their age, income, viewing habits, and subscription plans—we aim to develop a model that can predict which customers might leave the service. Such a model would allow the streaming service to better understand customer behavior and potentially prevent churn by targeting these at-risk customers with retention strategies like personalized content recommendations or promotions.

The second problem we tackle is identifying whether mushrooms are poisonous or edible using a mushroom dataset from UCI (mushrooms.csv). Mushroom toxicity prediction could be invaluable for those interested in foraging or anyone involved in mushroom farming and processing. Given a mushroom’s physical characteristics—such as color, odor, and shape—this model will predict its safety. If successful, it could support an app that provides users with warnings when a mushroom is predicted to be dangerous. Both predictive models aim to make accurate and reliable classifications, helping inform decisions in customer retention and health and safety.


## Methods

### Business Churn
To prepare the streaming data for modeling, we first checked for any missing values, as gaps in the data can affect the model’s performance. After addressing any missing values, we reset the indices to maintain an organized dataset. Then, we used a technique called Train-Test-Split Model Validation to divide the data into training and testing sets, with 90% of the data used for training the models and 10% reserved for testing. This allows us to see how well the models perform on unseen data, helping to identify if they are truly effective or just overfit to the training data. For preprocessing, we applied Z-score normalization to the continuous variables (e.g., age, income) to scale them so they all have the same range. Additionally, categorical variables (such as gender and plan type) were converted to numerical form using a method called One-Hot Encoding (or dummy variables), which helps the model interpret these categories without assigning them a specific order.

Once the data was prepared, we trained two models: Logistic Regression and Gradient Boosting Tree. Logistic Regression is a statistical model that predicts the probability of a binary outcome, in this case, whether a customer will "churn" or "stay." The model estimates the relationship between various input features (such as age, income, and months subscribed) and the likelihood of a customer leaving. Each feature is assigned a weight that reflects how strongly it influences the probability of churn. This model is especially useful for its simplicity and interpretability—each feature’s impact on churn is clear, making it easy to understand and communicate to stakeholders. However, its linear nature can limit its effectiveness with more complex relationships in data, so it may not perform as well as more sophisticated models when those are present.

Gradient Boosting Tree, on the other hand, is a more advanced machine learning model that builds a series of smaller decision trees to improve prediction accuracy. Each tree in the sequence learns from the mistakes of the previous ones, gradually refining the predictions. In this way, Gradient Boosting Tree is designed to capture more intricate patterns in the data, especially nonlinear relationships that may influence churn behavior in a streaming service. While it has the advantage of handling complex relationships, it is computationally intensive, requiring more processing time and resources. Gradient Boosting Trees can sometimes be prone to overfitting, especially when many trees are used, meaning they may perform well on the training data but not as reliably on new data.

After training, we evaluated their effectiveness by calculating key performance metrics for both the training and testing sets. These metrics included Accuracy, Recall, Precision, and ROC AUC. Accuracy gives an overall measure of correct predictions, while Recall focuses on the model's ability to identify actual churners. Precision measures how many of those predicted as churners actually churned, and ROC AUC assesses how well the model separates churners from non-churners. These metrics help us understand the model's strengths and weaknesses, determining whether it performs consistently across different sets of data.

### Mushroom Classification
For the mushroom classification task, we followed similar preparation steps. We started by checking for any missing values and resetting indices to ensure data consistency. Then, we divided the data using an 80/20 Train-Test-Split, where 80% of the data was used for training and 20% for testing. Like with the streaming data, we standardized continuous features and used One-Hot Encoding (or dummy variables) to convert categorical variables into a form suitable for modeling. For Categorical Naive Bayes specifically, we only included categorical or binary variables, as this model is designed to handle discrete data.

We built three models to classify mushrooms as either poisonous or edible: Categorical Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression. Categorical Naive Bayes is a simple probabilistic model that assumes each feature independently contributes to the likelihood of a mushroom being poisonous or edible. By calculating the probability of each feature independently (like color or cap shape) given a poisonous or edible label, it combines these probabilities to classify each mushroom. This model is computationally efficient and performs well with categorical data, though its assumption of feature independence may not always match real-world data interactions, potentially limiting accuracy.

K-Nearest Neighbors (KNN) is a classification model that bases its predictions on the characteristics of the “nearest” examples in the training data. For each new mushroom, the KNN model identifies the K mushrooms in the training data that are most similar to it, and it assigns the class (poisonous or edible) that is most common among those neighbors. KNN is easy to understand and interpret, as predictions are based directly on data similarity. However, it requires more computational power as the dataset grows since each new prediction involves comparing to every other data point.

Lastly, we applied Logistic Regression to the mushroom dataset to determine whether this model could also effectively identify toxic mushrooms based on categorical features. Like in the churn model, Logistic Regression estimates the relationship between each mushroom characteristic and its likelihood of being poisonous. While it’s a linear model and less flexible in capturing complex interactions, it is reliable and interpretable. We processed the mushroom data similarly to the churn data, encoding categorical features numerically and normalizing continuous features where needed.

After training, we calculated Accuracy, Recall, Precision, and ROC AUC for each model on both the training and testing sets. These metrics allowed us to compare the models’ performances and assess which one was most reliable for identifying poisonous mushrooms. By analyzing these metrics, we could determine which model was best suited for deployment, balancing accuracy with other important aspects like model complexity and interpretability. 


## Results

The models for both the streaming churn and mushroom classification tasks performed differently in terms of key performance metrics, reflecting the complexity of each prediction challenge. 

### Business Churn

<table>
  <tr>
    <td align="center">
      <strong>Logistic Regression</strong><br>
      <img src="https://github.com/user-attachments/assets/6f665ec9-c283-465d-b991-becb9f3b1c80" width="600"><br>
      <img src="https://github.com/user-attachments/assets/26542936-b10a-4558-bed0-d34c370733b0" width="600" height="400">
    </td>
    <td align="center">
      <strong>Gradient Boosting Tree</strong><br>
      <img src="https://github.com/user-attachments/assets/42c813fe-eb78-4db8-bc47-9f6ad0bb9070" width="600"><br>
      <img src="https://github.com/user-attachments/assets/d6a3d69b-fc5e-4a5d-866a-6ad6f3e877eb" width="600" height="400">
    </td>
  </tr>
</table>




For the streaming churn prediction, both the Logistic Regression and Gradient Boosting Tree models achieved around 73% accuracy on the test set. This level of accuracy means they correctly classified customers who would churn or remain subscribed about 73% of the time. However, both models had relatively low Recall (18% and 17%, respectively), indicating they struggled to capture a large portion of actual churners, which could be a limitation for a churn-prediction tool. Their Precision values, however, were around 61%, suggesting that when the models predicted churn, they were fairly reliable in doing so. The ROC AUC values, around 0.70, reveal moderate performance in distinguishing between churners and non-churners, slightly above random chance.

The performance metrics of these models suggest some potential issues with overfitting. The high accuracy and moderate ROC AUC, combined with low Recall, imply that the models may be fitting to patterns specific to the training data without generalizing well to unseen data, particularly in capturing churn cases. Gradient Boosting showed slight overfitting, as it had slightly higher train metrics compared to test metrics, which is common with complex models. For both models, moderate ROC AUC values and Recall indicate limited reliability in recognizing churn patterns, but if prioritizing simplicity, Logistic Regression is likely the better production choice due to its interpretability and lower computational cost. Gradient Boosting, while potentially more accurate with additional tuning, has a higher time complexity and is harder to explain to non-technical users.

For the CEO, the streaming model chosen (likely Logistic Regression for simplicity) could help predict customer churn and potentially guide retention strategies. Though its Recall is low, it can still offer insights into general patterns of churn. The CEO could use this information to flag potentially at-risk customers and explore customer engagement strategies or customized offers to reduce churn rates. 

For the ethical implications of the streaming churn model, ethical concerns are less severe but could arise if users are targeted or incentivized based on behavioral predictions, potentially raising privacy concerns. Ensuring transparency and user consent in the use of churn predictions could address these issues. 

<p align="center">
<img src="https://github.com/user-attachments/assets/72630bfc-b485-43a1-920e-e910d59688d6" width="600" height="400"/>

This graph shows how a customer’s favorite genre—Comedy, Drama, Romantic Comedy, Science Fiction, or Thriller—affects their likelihood to stop using the service (churn). For Comedy, Drama, and Romantic Comedy, the churn probabilities are similar and don’t vary much. This suggests that fans of these genres are generally consistent and less likely to churn. In contrast, Science Fiction and Thriller show more variation, with some customers in these genres having a higher risk of churning.

As a business owner, you could use this information to keep more customers engaged. For fans of Science Fiction and Thriller, you might need to engage them more frequently by promoting new or exclusive content in these genres. This could help maintain their interest, as these customers may be more likely to churn without fresh content. For customers who prefer Comedy, Drama, or Romantic Comedy, you may not need to do as much. Occasional updates or seasonal recommendations may be enough to keep them subscribed. Additionally, offering special deals or rewards to higher-risk groups like Science Fiction and Thriller fans—such as discounts or early access to content—could further reduce churn and encourage loyalty.

### Mushroom Classification

<table>
  <tr>
    <td align="center">
      <strong>Categorical Naive Bayes</strong><br>
      <img src="https://github.com/user-attachments/assets/669730ca-fa72-495f-a4b6-8d77ef4845bc" height="200">
    </td>
    <td align="center">
      <strong>K-Nearest Neighbors (KNN)</strong><br>
      <img src="https://github.com/user-attachments/assets/b2162d2f-2d4b-42cf-a87d-780db36c7a0e" height="200">
    </td>
    <td align="center">
      <strong>Logistic Regression</strong><br>
      <img src="https://github.com/user-attachments/assets/ed529bb3-92eb-4ade-ac16-5ed78c9a9d0b" height="200">
    </td>
  </tr>
</table>




For the mushroom classification task, all models—Categorical Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression—performed exceptionally well, with test accuracies close to 100%. Both KNN and Logistic Regression achieved perfect metrics (100%) in Recall, Precision, and ROC AUC on the test set, while Naive Bayes also performed with high accuracy and Recall (about 94.7% and 89.7%, respectively). The perfect scores from KNN and Logistic Regression suggest that these models might be overfit to the training data, especially KNN, which memorizes training samples and could fail with new, previously unseen data. However, these metrics also indicate that the models are very likely to be effective at predicting mushroom toxicity with high reliability in a real-world setting.

The metrics tell us that the mushroom classification models, particularly KNN and Logistic Regression, are highly trustworthy for making predictions. Their high Recall ensures they are very likely to catch all poisonous mushrooms, which is critical for safety. In contrast, Categorical Naive Bayes, with slightly lower Recall, might miss some poisonous cases, making it less ideal for a safety-focused application. Given these results, Logistic Regression would be the top choice for production due to its balance between simplicity, high accuracy, and interpretability.

For calibration, the mushroom models showed strong alignment between predicted probabilities and actual outcomes, particularly Logistic Regression, which is well-known for producing well-calibrated probabilities. This is essential, as well-calibrated probabilities allow us to make more accurate, probability-based decisions, which could be beneficial if the model was used in real-life settings, such as advising users on whether a mushroom is safe.

The mushroom classification model could be used directly in a mushroom detector app, where the Logistic Regression model’s high accuracy and simplicity would offer reliable results. Confidence in this model’s output is high given the excellent Recall and Precision metrics, although more confidence could come from validating the model on a larger, more varied mushroom dataset.

<p align="center">
<img src="https://github.com/user-attachments/assets/498b0106-7507-4ce2-ae74-f22481494bca">

For the ethical implications of the mushroom detector app, the primary concern is user safety; if the model misclassifies a poisonous mushroom as safe, the consequences could be severe. Thus, we must ensure the model’s high reliability and potentially implement safeguards like requiring a second model to confirm “safe” classifications. 



## Discussion/Reflection

This project highlighted the importance of selecting and carefully evaluating predictive models based on the needs of the problem at hand. For the churn prediction model, Logistic Regression provided basic insight into customer behavior but did not perform as well in detecting true churners compared to Gradient Boosting Tree. The mushroom classification task demonstrated how, with well-structured data, models like KNN and Logistic Regression can achieve very high accuracy. If given more time, I would consider adding more data features for churn prediction, perhaps examining how recent changes in streaming habits correlate with churn. This project demonstrated the power of data in making real-world decisions, whether helping companies retain customers or helping people make safe choices about mushroom foraging.


