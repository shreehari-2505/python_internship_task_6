# 🍷 K-Nearest Neighbors (KNN) — Multi-Class Classification on Wine Quality Dataset

## 🎯 Objective

This project is part of my AI & ML Internship – **Task 6**.

The goal is to build a **Multi-Class Classifier** using the **K-Nearest Neighbors (KNN)** algorithm to predict wine quality. The quality is categorized into three classes: **Low**, **Medium**, or **High**, based on the physicochemical properties of the Red Wine Quality dataset.

---

## 🛠️ Tools & Libraries Used

|Tool / Library|Purpose|
|---|---|
|**Python**|Core programming language|
|**Pandas**|Data loading, cleaning, and manipulation|
|**NumPy**|Numerical computations and array operations|
|**Matplotlib/Seaborn**|Data visualization for plots, heatmaps, and boundaries|
|**Scikit-learn**|Data preprocessing, model building, evaluation, and dimensionality reduction|
|**Google Colab**|Notebook execution environment|

---

## 🔄 Workflow – Step-by-Step Logic Flow

A text-based flowchart showing the entire process:

[Start]  
↓  
Load Red Wine Quality dataset (`pd.read_csv`)  
↓  
**Data Preprocessing:**

- Create a categorical target `quality_class` from the `quality` score (`pd.cut`)
- Separate features (X) and target (y)  
    ↓  
    Split data into Training (80%) and Test (20%) sets (`train_test_split`)  
    ↓  
    Standardize features to mean=0 and std=1 with `StandardScaler`  
    ↓  
    **Hyperparameter Tuning:**
- Loop through different K values to find the optimal number of neighbors
- Plot Accuracy vs. K value  
    ↓  
    Define and train the final KNN model with the best K (`KNeighborsClassifier`)  
    ↓  
    **Model Evaluation:**
- Predict on the test set (`.predict`)
- Evaluate performance using Accuracy Score and Confusion Matrix  
    ↓  
    **Visualization:**
- Use PCA to reduce features to 2D for plotting decision boundaries
- Plot class distribution and feature correlation heatmap  
    ↓  
    [End]

---

## 🔢 Steps Performed in Detail

#### 1. Data Loading & Initial Exploration

- **Dataset:** Red Wine Quality Data Set from the UCI Machine Learning Repository.
- **Loaded using:** `df = pd.read_csv(url, sep=';')`.
- Checked for unique values in the `quality` column.

#### 2. Data Preprocessing

- **Target Engineering:** The continuous `quality` column (ranging from 3 to 8) was converted into a categorical `quality_class` with three levels:
    - **0 (Low):** Quality score between 2 and 5
    - **1 (Medium):** Quality score of 5
    - **2 (High):** Quality score between 6 and 8
- **Feature/Target Separation:** The dataset was split into a feature matrix `X` (all columns except `quality` and `quality_class`) and a target vector `y` (`quality_class`).

#### 3. Feature Scaling

- All features in the training and test sets were standardized using `StandardScaler`. This is crucial for distance-based algorithms like KNN, as it prevents features with larger scales from disproportionately influencing the model.

#### 4. K-Selection and Model Training

- **Experimentation:** The KNN model was trained and evaluated on a range of K values (`[1, 3, 5, 7, 9]`) to find the optimal hyperparameter.
- **Training:** The final `KNeighborsClassifier` was fitted on the scaled training data (`X_train_scaled`, `y_train`).

#### 5. Predictions & Evaluation

- **Predictions:** The trained model was used to predict quality classes for the `X_test_scaled` data.
- **Computed:**
    - **Accuracy Score:** To measure the overall correctness of the model.
    - **Confusion Matrix:** To visualize model performance for each of the three classes (Low, Medium, High) and identify misclassifications.

#### 6. Visualization

- **Decision Boundaries:** Principal Component Analysis (PCA) was used to reduce the 11 features to 2 principal components. A new KNN model was trained on this 2D data to visualize the classification boundaries.
- **Class Distribution:** A `countplot` was plotted to show the imbalance in the dataset, with a majority of wines belonging to the 'Low' quality class.
- **Feature Correlation:** A heatmap was generated to show the correlation between each feature and the original `quality` score.

---

## 📙 Vocabulary of Functions & Commands Used

|Command / Function|Purpose|
|---|---|
|`pd.read_csv(path, sep)`|Reads a CSV file into a pandas DataFrame, specifying the separator.|
|`pd.cut(series, bins, labels)`|Bins values into discrete intervals, creating a categorical variable.|
|`df.drop(columns, axis)`|Removes specified rows or columns from a DataFrame.|
|`train_test_split()`|Splits arrays or matrices into random train and test subsets.|
|`StandardScaler()`|Standardizes features by removing the mean and scaling to unit variance.|
|`.fit_transform(data)`|Fits the scaler to data and then transforms it.|
|`.transform(data)`|Transforms data using a previously fitted scaler.|
|`KNeighborsClassifier(n_neighbors)`|Defines the KNN classification model with a specified K.|
|`.fit(X_train, y_train)`|Trains the model on the training set.|
|`.predict(X_test)`|Predicts class labels for new data.|
|`accuracy_score(y_true, y_pred)`|Computes the classification accuracy.|
|`confusion_matrix(y_true, y_pred)`|Computes a confusion matrix to evaluate classification accuracy.|
|`PCA(n_components)`|Reduces dimensionality of the data to the specified number of components.|
|`sns.heatmap()`|Plots rectangular data as a color-encoded matrix.|
|`sns.countplot()`|Shows the counts of observations in each categorical bin using bars.|
|`plt.plot()` / `plt.scatter()`|Creates line and scatter plots in `matplotlib`.|

---

## 💡 Key Insights

- The model's performance is highly dependent on the choice of **K**, with **K=1** yielding the highest accuracy of **~0.67**. This indicates that the nearest single neighbor is often the best predictor, but also makes the model susceptible to noise or outliers.
- **Feature scaling** is a critical and mandatory step for KNN, ensuring that all features contribute equally to the Euclidean distance calculation.
- The **confusion matrix** revealed that the model is fairly good at identifying 'Low' quality wines but struggles to differentiate between 'Medium' and 'High' quality, often misclassifying them.
- The dataset is **imbalanced**, with a significantly larger number of 'Low' quality samples. This can bias the model towards the majority class and explains some of the classification challenges.
- From the correlation heatmap, **`volatile acidity`** has the strongest negative correlation with quality, while **`sulphates`** and **`alcohol`** (not shown in the final heatmap but a known factor) have strong positive correlations
