Here’s a detailed `README.md` file for your GitHub repository based on the code you provided:

---

# Bigmart Sales Prediction

## Overview

This project focuses on predicting sales of items in various Bigmart outlets using the Bigmart Sales dataset. We perform exploratory data analysis (EDA), preprocess the data, and apply various regression models to predict `Item_Outlet_Sales`. We compare different models' performance, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, and Extra Trees Regressor.

## Data

The dataset `Train.csv` contains information about various items in different outlets. The columns include:

- `Item_Identifier`
- `Item_Weight`
- `Item_Fat_Content`
- `Item_Visibility`
- `Item_Type`
- `Item_MRP`
- `Outlet_Identifier`
- `Outlet_Establishment_Year`
- `Outlet_Size`
- `Outlet_Location_Type`
- `Outlet_Type`
- `Item_Outlet_Sales`

## Setup

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Data Preprocessing

1. **Load the Data:**

    ```python
    import pandas as pd
    df = pd.read_csv('Train.csv')
    ```

2. **Explore and Clean Data:**
    - Check unique values, missing values, and categorical attributes.
    - Fill missing values based on mean or mode.
    - Replace zeros in `Item_Visibility` with the mean value.
    - Combine and map values in `Item_Fat_Content` and create new features.

    ```python
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
    df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    ```

3. **Exploratory Data Analysis (EDA):**
    - Visualize distributions and correlations.
    - Use log transformation on the target variable `Item_Outlet_Sales`.

    ```python
    sns.distplot(df['Item_Weight'])
    sns.distplot(df['Item_MRP'])
    df['Item_Outlet_Sales'] = np.log(1 + df['Item_Outlet_Sales'])
    sns.distplot(df['Item_Outlet_Sales'])
    sns.countplot(df["Item_Fat_Content"])
    sns.countplot(df["Item_Type"])
    sns.countplot(df['Outlet_Establishment_Year'])
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    ```

4. **Encoding Categorical Features:**
    - Apply Label Encoding and One-Hot Encoding.

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
    cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
    ```

## Model Training

1. **Split Data:**

    ```python
    X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
    y = df['Item_Outlet_Sales']
    ```

2. **Define and Train Models:**

    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    def train(model, X, y):
        model.fit(X, y)
        pred = model.predict(X)
        cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
        cv_score = np.abs(np.mean(cv_score))
        print("Model Report")
        print("MSE:", mean_squared_error(y, pred))
        print("CV Score:", cv_score)
    
    # Linear Regression
    model = LinearRegression()
    train(model, X, y)
    
    # Ridge Regression
    model = Ridge()
    train(model, X, y)
    
    # Lasso Regression
    model = Lasso()
    train(model, X, y)
    
    # Decision Tree Regressor
    model = DecisionTreeRegressor()
    train(model, X, y)
    
    # Random Forest Regressor
    model = RandomForestRegressor()
    train(model, X, y)
    
    # Extra Trees Regressor
    model = ExtraTreesRegressor()
    train(model, X, y)
    ```

3. **Plot Coefficients and Feature Importances:**

    ```python
    coef = pd.Series(model.coef_, X.columns).sort_values()
    coef.plot(kind='bar', title="Model Coefficients")
    ```

## Conclusion

- **Best Model:** Out of the models tested, Linear Regression showed the best performance with the lowest cross-validation score.
- **Further Improvements:** Consider hyperparameter tuning and trying advanced models like XGBoost or CatBoost for better performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the `README.md` content as needed based on additional information or changes in your project.
