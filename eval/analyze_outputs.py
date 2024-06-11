import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# # Load the data from the JSON file
# with open('outputs/mtbenchmark.json') as f:
#     data = json.load(f)

# # Convert the data to a DataFrame
# df = pd.DataFrame(data)

# # Ensure that the score columns are numeric
# df['score_ads'] = pd.to_numeric(df['score_ads'], errors='coerce')
# df['score_ctrl'] = pd.to_numeric(df['score_ctrl'], errors='coerce')

# # Define the order and pattern of categories
# categories = ['Writing', 'Roleplay', 'Reasoning', 'Math', 'Coding', 'Extraction', 'STEM', 'Humanities']

# # Initialize lists to store the aggregated scores
# avg_scores_ads = []
# avg_scores_ctrl = []

# # Manually group and average the scores for each category
# for i in range(0, len(df), 10):
#     avg_scores_ads.append(df['score_ads'][i:i+10].mean())
#     avg_scores_ctrl.append(df['score_ctrl'][i:i+10].mean())

# # Add the first element at the end to close the radar chart

# # Number of variables
# num_vars = len(categories)

# categories.append(categories[0])
# avg_scores_ads.append(avg_scores_ads[0])
# avg_scores_ctrl.append(avg_scores_ctrl[0])

# # Compute angle of each axis
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles.append(angles[0])

# # Debug prints
# print(f"Categories: {categories}")
# print(f"Average Scores Ads: {avg_scores_ads}")
# print(f"Average Scores Ctrl: {avg_scores_ctrl}")
# print(f"Angles: {angles}")

# # Create the radar plot
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# # Draw one axe per variable
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)

# # Draw one axe per variable + add labels
# plt.xticks(angles, categories)

# # Draw ylabels
# ax.set_rscale('linear')
# plt.yticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'], color="grey", size=7)
# plt.ylim(0, 10)

# # Plot data
# ax.plot(angles, avg_scores_ads, linewidth=1, linestyle='solid', label='Ads Model')
# ax.fill(angles, avg_scores_ads, 'b', alpha=0.1)

# ax.plot(angles, avg_scores_ctrl, linewidth=1, linestyle='solid', label='Control')
# ax.fill(angles, avg_scores_ctrl, 'r', alpha=0.1)

# # Add a title
# plt.title('Model Performance Comparison')

# # Add a legend
# plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# plt.savefig('plots/mtbenchmark.pdf')


# Load data from the JSON files
with open('outputs/drop.json') as f:
    drop_data = json.load(f)

with open('outputs/mgsm.json') as f:
    mgsm_data = json.load(f)

# Convert data to DataFrames
drop_df = pd.DataFrame(drop_data)
mgsm_df = pd.DataFrame(mgsm_data)

# Calculate correctness rates
drop_correct_ads = drop_df['correct_ads'].mean()
drop_correct_ctrl = drop_df['correct_ctrl'].mean()

mgsm_correct_ads = mgsm_df['correct_ads'].mean()
mgsm_correct_ctrl = mgsm_df['correct_ctrl'].mean()

# Prepare data for plotting
data = {
    'Dataset': ['DROP', 'DROP', 'MGSM', 'MGSM'],
    'Model': ['Ads', 'Control', 'Ads', 'Control'],
    'Correctness Rate': [drop_correct_ads, drop_correct_ctrl, mgsm_correct_ads, mgsm_correct_ctrl]
}

df = pd.DataFrame(data)

# Create the bar plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Correctness Rate', hue='Model', data=df)
plt.title('Correctness Rates for DROP and MGSM Datasets')
plt.ylim(0, 1)
plt.ylabel('Correctness Rate')
plt.xlabel('Dataset')

# Save the plot as a PDF file
plt.savefig('plots/correctness_rates.pdf')

# Show the plot
plt.show()



