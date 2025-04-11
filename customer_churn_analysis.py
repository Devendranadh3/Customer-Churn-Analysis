# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv("C:\\Users\\hp\\Downloads\\Bank+Customer+Churn\\Bank_Churn.csv")
print("Dataset loaded successfully!\n")

# Show basic information
print(df.info())
print(df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Objective 1: Customer Churn Overview
# Bar chart
sns.countplot(x='Exited', data=df,color='brown')
plt.title('Customer Churn Distribution')
plt.show()

# Pie chart
churn_counts = df['Exited'].value_counts()
plt.pie(churn_counts, labels=['Stayed', 'Churned'], autopct='%1.1f%%',colors=['orange','#66b3ff'])
plt.title('Churn Percentage')
plt.show()

# Objective 2: Demographic Impact
# Gender vs Churn
sns.countplot(x='Gender', hue='Exited', data=df,palette='pastel')
plt.title('Churn by Gender')
plt.show()

# Geography vs Churn
sns.countplot(x='Geography', hue='Exited', data=df,palette='muted')
plt.title('Churn by Geography')
plt.show()

# Age vs Churn
sns.histplot(data=df, x='Age', hue='Exited', bins=20, kde=True)
plt.title('Age Distribution by Churn')
plt.show()

# Objective 3: Customer Behavior
# Tenure
sns.boxplot(x='Exited', y='Tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()

# Products
sns.countplot(x='NumOfProducts', hue='Exited', data=df,palette='deep')
plt.title('Products vs Churn')
plt.show()

# Active Member
sns.countplot(x='IsActiveMember', hue='Exited', data=df,palette='coolwarm')
plt.title('Active Member vs Churn')
plt.show()

# Objective 4: Financial Factors
# Credit Score
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score vs Churn')
plt.show()

# Balance
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Balance vs Churn')
plt.show()

# Estimated Salary
sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
plt.title('Salary vs Churn')
plt.show()

# Correlation Heatmap
corr = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']].corr()
sns.heatmap(corr, annot=True,cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# Pearson correlation
corr_val, _ = pearsonr(df['CreditScore'], df['Exited'])
print(f"Correlation between Credit Score and Churn: {corr_val:.2f}")
