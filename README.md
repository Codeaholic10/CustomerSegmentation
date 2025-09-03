# CustomerSegmentation
This project involves analyzing a transnational e-commerce dataset to identify distinct customer groups based on their purchasing behavior. The goal is to move beyond a "one-size-fits-all" marketing approach and enable data-driven, personalized engagement strategies.
Of course. Here is a comprehensive `README.md` file for your Customer Segmentation project. You can copy and paste this text into a file named `README.md` in your project's folder on GitHub.

-----

# Customer Segmentation using RFM Analysis & K-Means Clustering

## 📖 Project Overview

This project analyzes a transnational e-commerce dataset to identify distinct customer segments based on their purchasing behavior. By leveraging **RFM (Recency, Frequency, Monetary)** analysis and the **K-Means clustering** algorithm, this project moves beyond a "one-size-fits-all" marketing approach to provide data-driven, actionable insights for personalized engagement strategies.

The primary goal is to segment customers into meaningful groups, understand their characteristics, and provide tailored marketing recommendations to improve customer retention and maximize lifetime value.

-----

## 📊 Dataset

The project uses the **Online Retail Data Set** from the UCI Machine Learning Repository. This is a transnational dataset which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.

  * **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
  * **Format:** Excel file (`Online Retail.xlsx`)
  * **Attributes:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

-----

## ⚙️ Methodology

The analysis follows a structured workflow:

1.  **Data Cleaning & Preprocessing:**

      * Loaded the dataset and handled potential encoding issues.
      * Removed rows with missing `CustomerID` values.
      * Filtered out transactions representing returns (negative `Quantity`).
      * Cleaned any invalid data entries (e.g., `UnitPrice` of 0).

2.  **Feature Engineering with RFM:**

      * Calculated a `TotalPrice` column (`Quantity` \* `UnitPrice`).
      * Determined **Recency**, **Frequency**, and **Monetary** values for each unique customer.
      * Handled data skewness by applying a log transformation to the RFM metrics.

3.  **Clustering with K-Means:**

      * Scaled the RFM data using `StandardScaler` to ensure all features have equal weight.
      * Used the **Elbow Method** to determine the optimal number of clusters (k).
      * Applied the K-Means algorithm to assign each customer to a specific segment.

4.  **Visualization & Interpretation:**

      * Used **Principal Component Analysis (PCA)** to reduce the 3D RFM data into 2D for visualization.
      * Created a scatter plot to visualize the distinct customer segments.
      * Analyzed the mean RFM values of each cluster to build descriptive personas.

-----

## 💡 Results & Business Recommendations

The analysis successfully identified four distinct customer segments:

| Cluster ID | Persona             | Characteristics                                    | Marketing Recommendations                                                                                             |
| :--------- | :------------------ | :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| 0          | **Champions** | Low Recency, High Frequency, High Monetary         | Reward them with loyalty programs, early access to new products, and exclusive offers. Acknowledge their loyalty. |
| 1          | **At-Risk Customers** | High Recency, Moderate Frequency & Monetary        | Re-engage them with personalized promotions, special discounts, or "we miss you" campaigns to win them back.         |
| 2          | **New / Promising** | Low Recency, Low Frequency, Low Monetary           | Nurture the relationship through welcome offers, onboarding support, and product recommendations to foster growth.    |
| 3          | **Lost Customers** | Very High Recency, Very Low Frequency & Monetary | Target with low-cost campaigns or remove from active marketing lists to optimize spend.                           |

By tailoring marketing efforts to these segments, the business can allocate resources more effectively, leading to higher engagement rates and increased profitability.

-----

## 💻 Technologies Used

  * **Python 3.x**
  * **Pandas:** For data manipulation and analysis.
  * **NumPy:** For numerical operations and data transformation.
  * **Scikit-learn:** For scaling (`StandardScaler`), clustering (`KMeans`), and dimensionality reduction (`PCA`).
  * **Matplotlib & Seaborn:** For data visualization.
  * **Jupyter Notebook:** As the development environment.

-----

## ▶️ How to Run

1.  Clone this repository to your local machine:
    ```bash
    git clone <your-repository-url>
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
    ```
3.  Place the `Online Retail.xlsx` dataset in the project's root directory.
4.  Launch Jupyter Notebook or JupyterLab and open the project's `.ipynb` file.
    ```bash
    jupyter notebook
    ```
5.  Run the cells sequentially to reproduce the analysis.
