# Part 20 - Thompson Sampling

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

# Import dataset
dataset = pd.read_csv('datasets/Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling algorithm
# Number of rounds
N = 10000
# Number of unique ads
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
# Loop through dataset N times
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # Assigns a random value the probability that an ad will increase their reward score
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # Picks the ad with the highest value
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    # Puts ad into ads_selected list
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    # Updates reward score of the ad, either 1 or 0
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    # Total reward score of the ad
    total_reward = total_reward + reward

# Visualizing the histogram of the number of times an ad was selected
plt.figure('Histogram of ads selections')
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

# Show plot
plt.show()