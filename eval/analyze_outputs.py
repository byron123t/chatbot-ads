import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


dfs = []
for i in range(10):
    with open(f'outputs/mtbenchmark{i}.json') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        df['try'] = f'Try {i+1}'
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

avg_scores_ads_4omini = []
avg_scores_ctrl_4omini = []
avg_scores_ads_3 = []
avg_scores_ctrl_3 = []
avg_scores_ads_4o = []
avg_scores_ctrl_4o = []
categories = ['Writing', 'Roleplay', 'Reasoning', 'Math', 'Coding', 'Extraction', 'STEM', 'Humanities']

for category in categories:
    avg_scores_ads_4omini.append(df[(df['category'] == category) & (df['mode'] == 'interest-based') & (df['model'] == 'gpt-4o-mini')]['score'].mean())
    avg_scores_ctrl_4omini.append(df[(df['category'] == category) & (df['mode'] == 'control') & (df['model'] == 'gpt-4o-mini')]['score'].mean())
    avg_scores_ads_3.append(df[(df['category'] == category) & (df['mode'] == 'interest-based') & (df['model'] == 'gpt-3.5-turbo')]['score'].mean())
    avg_scores_ctrl_3.append(df[(df['category'] == category) & (df['mode'] == 'control') & (df['model'] == 'gpt-3.5-turbo')]['score'].mean())
    avg_scores_ads_4o.append(df[(df['category'] == category) & (df['mode'] == 'interest-based') & (df['model'] == 'gpt-4o')]['score'].mean())
    avg_scores_ctrl_4o.append(df[(df['category'] == category) & (df['mode'] == 'control') & (df['model'] == 'gpt-4o')]['score'].mean())

all_scores_ads = []
all_scores_ctrl = []

num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

avg_scores_ads_4omini += avg_scores_ads_4omini[:1]
avg_scores_ctrl_4omini += avg_scores_ctrl_4omini[:1]
avg_scores_ads_3 += avg_scores_ads_3[:1]
avg_scores_ctrl_3 += avg_scores_ctrl_3[:1]
avg_scores_ads_4o += avg_scores_ads_4o[:1]
avg_scores_ctrl_4o += avg_scores_ctrl_4o[:1]

print(avg_scores_ads_4omini)
print(avg_scores_ctrl_4omini)
print(sum(avg_scores_ads_4omini[:-1]) / (len(avg_scores_ads_4omini) - 1))
print(sum(avg_scores_ctrl_4omini[:-1]) / (len(avg_scores_ctrl_4omini) - 1))

sns.set_theme(font_scale=1.5, style='whitegrid', palette='bright')
fig, ax = plt.subplots(figsize=(10, 11), subplot_kw=dict(polar=True))

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories)

ax.set_rscale('linear')
plt.yticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'], color="grey", size=20)
plt.ylim(0, 10)

ax.plot(angles, avg_scores_ads_4omini, color='#66c2a5', linewidth=2, linestyle='solid', label='Ads GPT-4o-mini')
ax.fill(angles, avg_scores_ads_4omini, '#66c2a5', alpha=0.1)

ax.plot(angles, avg_scores_ctrl_4omini, color='#fc8d62', linewidth=2, linestyle='dashed', label='Control GPT-4o-mini')
ax.fill(angles, avg_scores_ctrl_4omini, '#fc8d62', alpha=0.1)

# ax.plot(angles, avg_scores_ads_3, color='#7570b3', linewidth=2, linestyle='solid', label='Ads gpt-3.5-turbo')
# ax.fill(angles, avg_scores_ads_3, '#7570b3', alpha=0.07)

# ax.plot(angles, avg_scores_ctrl_3, color='#1b9e77', linewidth=2, linestyle='dashed', label='Control gpt-3.5-turbo')
# ax.fill(angles, avg_scores_ctrl_3, '#1b9e77', alpha=0.07)

# ax.plot(angles, avg_scores_ads_4o, color='#d95f02', linewidth=2, linestyle='solid', label='Ads gpt-4o')
# ax.fill(angles, avg_scores_ads_4o, '#d95f02', alpha=0.07)

# ax.plot(angles, avg_scores_ctrl_4o, color='#7570b3', linewidth=2, linestyle='dashed', label='Control gpt-4o')
# ax.fill(angles, avg_scores_ctrl_4o, '#7570b3', alpha=0.07)


plt.legend(loc='lower right', bbox_to_anchor=(1.14, -0.205))

plt.savefig('plots/mtbenchmark.pdf')


datasets = {
    'DROP': ['outputs/drop0.json', 'outputs/drop1.json', 'outputs/drop2.json', 'outputs/drop3.json', 'outputs/drop4.json', 'outputs/drop5.json', 'outputs/drop6.json', 'outputs/drop7.json', 'outputs/drop8.json', 'outputs/drop9.json'],
    'MGSM': ['outputs/mgsm0.json', 'outputs/mgsm1.json', 'outputs/mgsm2.json', 'outputs/mgsm3.json', 'outputs/mgsm4.json', 'outputs/mgsm5.json', 'outputs/mgsm6.json', 'outputs/mgsm7.json', 'outputs/mgsm8.json', 'outputs/mgsm9.json'],
    'MMLU': ['outputs/mmlu0.json', 'outputs/mmlu1.json', 'outputs/mmlu2.json', 'outputs/mmlu3.json', 'outputs/mmlu4.json', 'outputs/mmlu5.json', 'outputs/mmlu6.json', 'outputs/mmlu7.json', 'outputs/mmlu8.json', 'outputs/mmlu9.json'],
    'MATH': ['outputs/math0.json', 'outputs/math1.json', 'outputs/math2.json', 'outputs/math3.json', 'outputs/math4.json', 'outputs/math5.json', 'outputs/math6.json', 'outputs/math7.json', 'outputs/math8.json', 'outputs/math9.json'],
    'HE': ['outputs/humaneval0.json', 'outputs/humaneval1.json', 'outputs/humaneval2.json', 'outputs/humaneval3.json', 'outputs/humaneval4.json', 'outputs/humaneval5.json', 'outputs/humaneval6.json', 'outputs/humaneval7.json', 'outputs/humaneval8.json', 'outputs/humaneval9.json'],
    'GPQA': ['outputs/gpqa0.json', 'outputs/gpqa1.json', 'outputs/gpqa2.json', 'outputs/gpqa3.json', 'outputs/gpqa4.json', 'outputs/gpqa5.json', 'outputs/gpqa6.json', 'outputs/gpqa7.json', 'outputs/gpqa8.json', 'outputs/gpqa9.json'],
}



# Function to calculate correctness rate
def calculate_correctness_rate(data, model_name):
    correct_responses = [item for item in data if item['mode'] == model_name and item['correct']]
    total_responses = [item for item in data if item['mode'] == model_name]
    if len(total_responses) == 0:
        return 0  # To avoid division by zero
    return len(correct_responses) / len(total_responses)

models = ['Ads', 'Control']
modes = ['interest-based', 'control']

plot_data = {
    'Dataset': [],
    'Mode': [],
    'Correctness Rate': [],
    'Try': []
}

for dataset_name, files in datasets.items():
    avg_scores_temp_ads = []
    avg_scores_temp_ctrl = []
    for i, file in enumerate(files):
        with open(file) as f:
            dataset_data = json.load(f)
        for model, mode in zip(models, modes):
            correctness_rate = calculate_correctness_rate(dataset_data, mode)
            plot_data['Dataset'].append(dataset_name)
            plot_data['Mode'].append(model)
            plot_data['Correctness Rate'].append(correctness_rate)
            plot_data['Try'].append(f'Try {i + 1}')
            if model == 'Ads':
                avg_scores_temp_ads.append(correctness_rate)
            else:
                avg_scores_temp_ctrl.append(correctness_rate)
    print(f'{dataset_name} Ads: {np.mean(avg_scores_temp_ads)}')
    print(f'{dataset_name} Control: {np.mean(avg_scores_temp_ctrl)}')
    
df = pd.DataFrame(plot_data)

sns.set_theme(font_scale=1.5, style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Correctness Rate', hue='Mode', data=df, errorbar='sd')
plt.title('Correctness Rates for LLM Datasets')
plt.ylabel('Correctness Rate')
plt.xlabel('Dataset')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('plots/correctness_rates.pdf')
