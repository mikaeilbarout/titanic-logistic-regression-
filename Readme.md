# Titanic Survival Prediction

Predicting which passengers survived the Titanic using **Logistic Regression (scikit-learn)**.

## Dataset

The classic Titanic passenger data (`titanic_train.csv`, `titanic_test.csv`), with features such as passenger class, sex, age, number of siblings/spouses and parents/children aboard, fare and port of embarkation.

## What the notebook does

1. **Explores the data** with Seaborn: survival by sex, by passenger class, the age distribution and fares
2. **Finds missing values** with a heatmap
3. **Fills missing ages** using the typical age of each passenger class (a box plot showed that first-class passengers tended to be older)
4. **Drops** the `Cabin` column (mostly empty) and the remaining rows with missing values
5. **Encodes** `Sex` and `Embarked` with one-hot encoding and drops text columns (`Name`, `Ticket`)
6. **Splits** the data 70/30 into training and test sets
7. **Trains** a Logistic Regression model
8. **Evaluates** it with a classification report and a confusion matrix

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Precision (survived) | 0.81 |
| Recall (survived) | 0.63 |

The model is better at spotting passengers who did not survive than those who did (lower recall for survivors).

## What I learned

- How to fill missing values in a smarter way than a single overall mean, by using a related feature (passenger class)
- Why categorical columns need encoding before a model can use them
- How to read precision, recall and a confusion matrix, not just accuracy

## How to run

```bash
pip install -r requirements.txt
jupyter notebook "titanic Logistic Regression.ipynb"
```
