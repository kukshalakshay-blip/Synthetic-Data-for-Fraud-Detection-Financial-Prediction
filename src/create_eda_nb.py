import nbformat as nbf

def main():
    nb = nbf.v4.new_notebook()

    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 02. Exploratory Data Analysis\n\nIn this notebook, we explore the Credit Card Fraud dataset downloaded from Kaggle to observe class imbalances and patterns."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='whitegrid')"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/raw/creditcard.csv')\ndf.head()"),
        nbf.v4.new_code_cell("df.info()"),
        nbf.v4.new_markdown_cell("## 2. Check for Class Imbalance"),
        nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.countplot(data=df, x='Class')\nplt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')\nplt.yscale('log')\nplt.show()\n\nprint('Class counts:')\nprint(df['Class'].value_counts())\nprint('\\nFraud percentage: {:.3f}%'.format(100 * df['Class'].sum() / len(df)))"),
        nbf.v4.new_markdown_cell("## 3. Visualize Features"),
        nbf.v4.new_code_cell("plt.figure(figsize=(10,6))\nsns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', kde=True)\nplt.title('Distribution of Transaction Amounts for Fraud')\nplt.legend()\nplt.show()")
    ]

    with open('notebooks/02_exploratory_data_analysis.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    main()
