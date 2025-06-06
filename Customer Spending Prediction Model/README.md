# Customer Spending Prediction Model

## Tech Stack
### Language & Environment
- Python
- Google Colab

### Libraries Used
- `pandas`, `numpy` – Data manipulation
- `plotnine` – Data visualization (ggplot-style)
- `scikit-learn` –
  - **Modeling**: `LinearRegression`, `LassoCV`, `RidgeCV`, `ElasticNetCV`
  - **Preprocessing**: `StandardScaler`, `PolynomialFeatures`, `SplineTransformer`
  - **Validation**: `train_test_split`, `KFold`, `LeaveOneOut`
  - **Evaluation**: `mean_squared_error`, `r2_score`, `mean_absolute_error`, `mean_absolute_percentage_error`
  - **Pipelines**: `make_pipeline`, `Pipeline`, `make_column_transformer`
- `%matplotlib inline` – For plotting inside notebooks

## Introduction
The objective of this analysis is to predict the average annual spending of customers at a clothing store based on various demographic and behavioral factors for a clothing store dataset (boutique.csv). Specifically, we aim to forecast the variable amount_spent_annual, which represents the average amount a customer spends per year at the store. The variables available include the following:
- `age`: current age of customer
- `height_cm`: self-reported height converted to centimeters
- `waist_size_cm`: self-reported waist size converted to centimeters
- `inseam_cm`: self-reported inseam (measurement from crotch of pants to floor) converted to centimeters
- `test_group`: whether or not the customer is in an experimental test group that gets special coupons once a month. 0 for no, 1 for yes.
- `salary_self_report_in_k`: self-reported salary of customer, in thousands
- `months_active`: number of months customer has been part of the clothing store's preferred rewards program
- `num_purchases`: the number of purchases the customer has made (a purchase is a single transaction that could include multiple items)
- `amount_spent_annual`: the average amount the customer has spent at the store per year
- `year`: the year the data was collected

By employing both Linear Regression and Polynomial Regression models, we can assess how different factors influence spending habits. Linear Regression looks for a straight-line relationship between our predictors (the factors) and spending, while Polynomial Regression can capture more complex relationships. If our models work well, they could help the clothing store better understand what influences how much customers spend. For instance, if we find out that customers with higher salaries and longer membership in the rewards program spend more, the store can focus on targeting these customers with specific marketing strategies, promotions, or personalized offers. Overall, these insights could lead to improved customer satisfaction and increased sales for the store.


## Methods 

Before building the models, we took several important steps to prepare our data. First, we looked for any missing values in the dataset. When we found rows with missing data, we removed them to ensure our models would be accurate and reliable. Next, we split the data into two sets: an 80% training set and a 20% testing set. The training set is used to teach the model, while the testing set evaluates how well the model performs on new, unseen data. This split is crucial for understanding how well our model can make predictions in real-world scenarios.

Since some of our variables, like gender, are categorical (meaning they represent categories rather than numbers), we used a technique called one-hot encoding (or dummy variables). This means we created separate columns for each category (e.g., male, female, nonbinary) so the model can understand and process these variables more effectively. We also standardized our numerical features, like age and salary, using a method called z-score normalization. This adjusts the values so they have a mean of zero and a standard deviation of one, ensuring that all variables contribute equally to the model and preventing any single variable from skewing the results.

The Linear Regression model looks for a straight-line relationship between the customer characteristics (the predictors) and their annual spending (the target variable). After training the model with our standardized training data, it analyzes the patterns in the data to find the best coefficients that minimize the difference between what customers actually spent and what the model predicted. To evaluate how well the model performs, we use metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²). MSE measures the average squared difference between predicted and actual spending, where lower values indicate better performance. MAE shows the average absolute difference, providing another perspective on accuracy, while R² tells us how much of the spending variability can be explained by our predictors; values closer to 1 indicate a stronger model fit.

The Polynomial Regression model takes it a step further by allowing for more complex, non-linear relationships between the predictors and annual spending. Instead of just fitting a straight line, this model can fit curves that better capture how various factors influence spending. To build this model, we transformed our training data into polynomial features, creating new variables that represent interactions and powers (n=2) of the original predictors. We then trained a Linear Regression model using these new features. Just like with the Linear Regression model, we evaluated the Polynomial Regression model using MSE, MAE, and R². Comparing these metrics to the Linear Regression results helps us see if the polynomial model offers significant improvements.


## Results 

<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>Linear Regression</strong><br>
        <img src="https://github.com/user-attachments/assets/218b2dd6-0e57-486c-ad12-2ff4b10fc83d" width="400">
      </td>
      <td align="center">
        <strong>Polynomial Regression (n=2)</strong><br>
        <img src="https://github.com/user-attachments/assets/8dce94c1-1947-4ea4-80b4-fee9db33abd9" width="400">
      </td>
    </tr>
  </table>
</div>



When we look at how well our models performed in predicting customer spending, we see that both the Linear Regression and Polynomial Regression models offer valuable insights. The linear regression model gave us a training Mean Squared Error (MSE) of 13,131.23 and a Mean Absolute Error (MAE) of 90.52. Its R-squared (R²) value was about 0.51, which means it explains roughly 51% of the differences in how much customers spend each year. This shows that the model has some predictive power, but it also indicates that there is room for improvement; we want the R² value closer to 1. Since the MSE and MAE values for both the training and testing sets are fairly close, we can conclude that the model is generalizing well, meaning it is not overly complex or too simplistic.

In contrast, the polynomial regression model performed much better. It achieved a training MSE of 3,350.92 and a training MAE of 46.26, with an R² value of around 0.88. This means it explains about 88% of the variability in annual spending, which is significantly better than the linear model. The lower MSE and MAE indicate that the predictions made by the polynomial model are much closer to what customers actually spent. Additionally, the fact that the performance metrics for both the training and testing datasets are similar indicates that this model is successfully identifying how various factors affect spending, without becoming overly complicated.

Using Polynomial Features helped enhance the polynomial model's performance. The substantial improvement we observed—reflected in the lower MSE and higher R² values—suggests that there are non-linear relationships and interactions among the customer characteristics that the linear model couldn't capture. By including these polynomial features, the model can better represent the complexities of customer spending. Based on the strong results from the polynomial model, we can trust it and confidently recommend it for use by the store. Its ability to explain spending patterns and its accuracy suggests it can provide valuable insights into how customers spend their money. However, there are a few important factors to keep in mind. First, the model's effectiveness depends on the quality of the data we have. If there are any biases or errors in the customer data, it could lead to unreliable predictions. Additionally, customer behavior can change over time due to factors like economic shifts or fashion trends, so it’s essential to regularly update the model with new data to keep it relevant. 

### Question 1: Does being in the experimental test_group actually increase the amount a customer spends at the store? Is this relationship different for the different genders?

<p align="center">
<img src="https://github.com/user-attachments/assets/e9133919-4fe9-47c0-9700-e3e1fd416656" width="600" height="400"/>

Looking at the box plot, it seems like people in the test group are spending more money than those who aren't. The line in the middle of the box shows the average amount spent, and in this case, the test group’s line is higher. This means that most people in the test group are buying more stuff at the store. There are also some people in the test group who spent a lot more than the average (outliers). This could mean that being part of the test group helps customers spend more money overall, so it looks like being in that group does make a difference in how much they spend.

<p align="center">
<img src="https://github.com/user-attachments/assets/852be408-36c6-472c-958f-b6d103979532" width="600" height="400"/>

This box plot shows how much different genders spend if they are part of the test group or not. Each color in the box plot represents a different gender, making it easier to compare. When we look closely, we can see that one gender might be spending more money than the other in both groups. Women tend to be spending more compared to the other genders (regardless if they are in the test group or not). The size of the boxes tells us how much people’s spending varies, and we can see that some groups have a bigger range, meaning some people spend a lot more than others.

### Question 2: In this dataset, is there a relationship between salary and height? Is it different for the different genders?

<p align="center">
<img src="https://github.com/user-attachments/assets/b31ddce8-620f-4622-be9f-9add8b617d15" width="600" height="400"/>

The first plot shows the overall relationship between salary and height for everyone in the dataset. From the graph, there is a slight negative trend, as indicated by the red line sloping downward. This suggests that as height increases, salary tends to decrease a little. However, this trend is not very strong, as the data points are scattered widely around the line. The spread of the points shows that there is a lot of variation in salary that isn’t explained by height, meaning height does not seem to have a significant impact on how much people earn. While the downward slope of the line shows a minor pattern, the relationship between height and salary is weak in this dataset.

<p align="center">
<img src="https://github.com/user-attachments/assets/40b00385-031f-47a5-a241-250553cfd97a" width="600" height="400"/>

The second set of graphs breaks down the relationship between salary and height by gender, and the trends vary across the different groups. For men, the red trend line slopes upward slightly, indicating a small positive relationship between height and salary; taller men tend to have higher salaries. In contrast, for women, the red line slopes downward, showing that taller women tend to have slightly lower salaries. For nonbinary individuals and those classified as "other," the red lines are nearly flat, which means there is no noticeable relationship between height and salary for these groups. These gender-based differences suggest that while height may influence salary for men and women, it does not have the same effect across all genders. Overall, the relationship between salary and height is weak and varies by gender.


## Discussion/Reflection

Doing these analyses taught me a lot about how important it is to choose the right model to understand customer spending. I was impressed by how much better the polynomial regression model performed compared to the linear model, which showed me that customer behavior can be more complex than just simple trends. I also learned that having accurate and clean data is crucial for making reliable predictions.
If I were to repeat this analysis in the future, I would focus on gathering more detailed information about customers, like their shopping habits or favorite products, to help improve the model’s accuracy. Additionally, I would use charts and graphs to present the findings more clearly, making it easier for others to understand what’s uncovered.

