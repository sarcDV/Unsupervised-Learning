# Importing the required libraries
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Loading and exploring the data
data = pd.read_excel('Online_Retail.xlsx')
data.head()

# Data preprocessing
data['Description'] = data['Description'].str.strip() # Removing spaces from the beginning and end of the strings
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True) # Dropping the rows without any invoice number
data['InvoiceNo'] = data['InvoiceNo'].astype('str') # Converting the invoice number column to string type
data = data[~data['InvoiceNo'].str.contains('C')] # Removing the credit transactions (those with invoice numbers containing 'C')

# Transforming the data into a basket format for each country
basket = (data[data['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Encoding the data with 0 and 1 values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# Applying Apriori algorithm with min_support=0.07
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

# Generating association rules with min_confidence=0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Printing the rules
rules.head()

"""
Apriori algorithm is an algorithm for frequent itemset mining 
and association rule learning over relational databases. 
It is based on the concept that a subset of a frequent itemset must also be a frequent itemset. 
A frequent itemset is an itemset that has a support value greater than a threshold value. 
Support value is the fraction of transactions that contain an itemset.

Apriori algorithm works in two steps: join and prune. It proceeds by identifying the frequent 
individual items in the database and extending them to larger and larger itemsets as long as 
those itemsets appear sufficiently often in the database. 
The join step generates candidate itemsets of size k by joining frequent itemsets of size k-1 with themselves. 
The prune step eliminates the candidate itemsets that have infrequent subsets using the Apriori property. 
This process is repeated until no more candidate itemsets are generated or no more frequent itemsets are found.

Apriori algorithm uses frequent itemsets to generate association rules, 
which are implications of the form X -> Y, where X and Y are disjoint itemsets. 
Association rules are evaluated by two metrics: confidence and lift. 
Confidence is the fraction of transactions that contain both X and Y among those that contain X. 
Lift is the ratio of confidence to the expected confidence, which is 
the fraction of transactions that contain Y in the whole database. 
A high confidence value indicates a strong association between X and Y, 
while a high lift value indicates that Y is more likely to occur when X occurs than by chance.
"""
