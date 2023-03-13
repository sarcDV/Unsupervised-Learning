# Importing libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Creating a sample dataset
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# Encoding the transactions into a binary matrix
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Applying the fpgrowth algorithm with min_support=0.6
frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)

# Printing the results
print(frequent_itemsets)

"""
FP-growth is an algorithm for frequent pattern mining, 
which finds frequent patterns or associations from data sets. 
It is an improved version of the Apriori algorithm, which scans the data set only twice. The basic steps of FP-growth are:

1.Build the FP-tree, a compact representation of the transactions that links similar items together.
2.Mine frequent itemsets from the FP-tree by following the links and counting the support2.
"""
## pip install pyfim
# Importing libraries
import pandas as pd
from sklearn.datasets import load_iris
from fim import fpgrowth

# Loading the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Binarizing the features using a threshold of 3
df = df.applymap(lambda x: 1 if x > 3 else 0)

# Converting the dataframe into a list of transactions
transactions = df.values.tolist()

# Applying the fpgrowth algorithm with min_support=0.6
frequent_itemsets = fpgrowth(transactions, supp=60)

# Printing the results
print(frequent_itemsets)
