# Data Science Mini Projects Portfolio

A collection of beginner-to-intermediate machine learning and data analysis projects completed in Google Colab.  
Each project demonstrates practical data cleaning, visualization, and model-building skills.

---

## Projects Overview

### Titanic Survival Prediction
**Goal:** Predict passenger survival using logistic regression, decision tree, random forest, and KNN.  
**Key Concepts:** Feature encoding, model comparison, confusion matrix, feature importance.

### House Prices Prediction
**Goal:** Predict house sale prices using regression models.  
**Key Concepts:** Handling missing data, feature scaling, linear regression, RMSE evaluation.

### Mall Customers Segmentation
**Goal:** Segment customers using K-Means clustering.  
**Key Concepts:** Elbow method, cluster visualization (Age vs Income vs Spending Score).

### Airline Tweets Sentiment Analysis
**Goal:** Analyze tweet sentiment using Hugging Face’s `cardiffnlp/twitter-roberta-base-sentiment` transformer model.  
**Key Concepts:** Text preprocessing, tokenization, model evaluation, real-world NLP application.

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Transformers, K-Means, Logistic Regression  
- **Environment:** Google Colab  
- **Data Source:** [Kaggle](https://www.kaggle.com)

---

## How to Use

1. Open the corresponding Google Colab notebook:  
   - Titanic Survival Prediction  
   - House Prices Prediction  
   - Mall Customers Segmentation
   - Airline Tweets Sentiment Analysis

2. Upload your **Kaggle API key** (see below) to automatically load datasets.  

3. Run the cells in order — each section includes explanations and outputs.

---

## How to Get Your Kaggle API Key

1. Go to your Kaggle account → **Profile picture → Settings → API → Create New API Token**.  
2. A file named `kaggle.json` will be downloaded automatically.  
3. In Google Colab, upload it using:

   ```python
   from google.colab import files
   files.upload()  # Then choose your kaggle.json
4. Move it to the correct location

    ```python
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
5. You can now use the Kaggle API to download datasets in any notebook.

---

## Results Summary

| **Project**                           | **Model(s) Used**                                                            | **Best Accuracy / Score**             | **Key Insights**                                                                                                                                |
| ------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Titanic Survivability Prediction**  | Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN) | **Random Forest – 82% accuracy**      | Sex, Fare, and Age were the strongest predictors of survival. Random Forest slightly outperformed other models.                                 |
| **House Prices Prediction**           | Linear Regression, Random Forest                                             | **Random Forest – RMSE ≈ 30 000**         | Random Forest captured non-linear relationships better than Linear Regression. Living Area and Overall Quality were the top predictors.         |
| **Mall Customers Segmentation**       | K-Means Clustering                                                           | **5 optimal clusters (Elbow Method)** | Segments formed around combinations of Income and Spending Score, identifying high-value and budget-conscious customer groups.                  |
| **Airline Tweets Sentiment Analysis** | Hugging Face DistilBERT Transformer                                          | **Accuracy ≈ 76% on test set**        | Model effectively classified sentiments (Positive, Neutral, Negative). Most tweets were Negative, focusing on service delays and cancellations. |

---

## Notes

- Each notebook is fully commented and includes visualizations.
- All datasets come from publicly available Kaggle sources.
- For demonstration purposes, smaller subsets of large datasets may be used in Colab.
   
