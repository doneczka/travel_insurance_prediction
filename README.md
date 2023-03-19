# travel_insurance_prediction

## **GOAL** – prediction of which customer is going to buy insurance.

-	Find personal traits of clients who are likely to buy an insurance
-	Fine-tune the model to get the best results

## **Methods and techniques:**

-	EDA as a starting point for drawing conclusions about clients characteristics
-	Visualization techniques for data distributions
-	Statistical inference to check statistical significance in difference between people who buy or do not buy insurance
-	Basic feature engineering 
-	Logistic Regression/SVC/k-NN/Decision Tree/Random Forest/Naïve Bayes for modeling
-	GridSearchCV/RandomizedSearchCV for hyperparameter tuning (depending on type of the model)
-	Ensembling techniques (voting classifier, bagging)
-	Feature importance with permutation_importance and shap library

## **Conclusions:**
I achieved: 74,78% of accuracy for rbf-SVM with C=0.6 and gamma=0.1 and 78,48% of accuracy for Decision Tree with max_depth=3.

Based on these models I could also make conclusions what features have the highest impact if the customer buy the insurance or not, which is:
- annual income
- number of family members
- fact if person travelled abroad or not.

Taking a look at what features were important in the decision tree, higher annual income indicates that customer more likely buy an insurance.

## **CALL TO ACTION:**

2 strategies to deal with this:
- target campaigns to more luxury travel offers so more people travelling at the high cost would be exposed to the agency offer
- make a special discount for people who buy low-fee trips, to make an insurance affordable for them.

Age was important feature too, but the range of age of customers was only 25-35. These people have similar activites and are on more or less the same life stage, so it would be hard to consider special offers for people based solely on age.

Family Members is another important factor. People who have bigger families more likely buy the package. Strategies:
- more offers for family holidays
- special offer for people who do not have a big family.



