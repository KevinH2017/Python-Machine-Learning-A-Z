# Part 19 - Upper Confidence Bound (UCB)

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import math

# Import dataset
dataset = pd.read_csv('datasets/Ads_CTR_Optimisation.csv')

# Implementing Upper Confidence Bound algorithm
# Number of rounds to loop through 
N = 10000
# Number of unique ads
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# Loop through dataset N times
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        # If ad is selected, creates confidence score by number of times it was selected
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # Apply highest value to declare maximum upper bound when finished with ads
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    # Assigns ad the confidence score calculated above
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualizing the histogram of the number of times an ad was selected
plt.figure('Histogram of ads selections')
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

# Show plot
plt.show()