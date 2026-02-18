# Part 17 - Apriori Algorithm

# Import libraries
import pandas as pd
from apyori import apriori

def inspect(results):
    '''
    Takes the results from the apriori function and extracts 
    the relevant information to display it in a readable format.
    
    :param results: The results from the apriori function.
    :return: A list of tuples containing the left-hand side, 
            right-hand side, support, confidence, and lift of the 
            association rules.
    '''
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

# Import dataset
dataset = pd.read_csv('datasets/Market_Basket_Optimisation.csv', header=None)

# Data preprocessing
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset.columns))])

# Training Apriori model on the dataset
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Visualising the results
# Displays the results from the apriori function
results = list(rules)
print(results)

# Puts results into Pandas DataFrame
resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(resultsinDataFrame)

# Displays the results sorted by descending Lift
desc_lift = resultsinDataFrame.nlargest(n=10, columns='Lift')
print(desc_lift)
