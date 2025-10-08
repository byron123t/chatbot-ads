import json
import pandas as pd
import matplotlib.pyplot as plt

# === Replace this with loading your full JSON data ===
data = json.load(open("outputs/temperature_mtbenchmark0.json", 'r'))
# === Normalize into DataFrame ===
df = pd.json_normalize(data)

# Ensure product_name_present is numeric (True/False -> 1/0)
df['product_name_present'] = df['product_name_present'].astype(int)

# Clean ad_subtlety_score: coerce to numeric, treat -1 as missing
df['ad_subtlety_score'] = pd.to_numeric(df['ad_subtlety_score'], errors='coerce')
df.loc[df['ad_subtlety_score'] < 0, 'ad_subtlety_score'] = pd.NA

# Group by top_p and compute metrics
grouped = df.groupby('temperature').agg(
    avg_score=('score', 'mean'),
    avg_ad_subtlety_score=('ad_subtlety_score', 'mean'),
    product_name_presence_rate=('product_name_present', 'mean'),
    count=('score', 'size')
).reset_index()

# Optional: inspect the aggregated table
print(grouped)

# === Plotting ===
plt.figure()
plt.plot(grouped['temperature'], grouped['avg_score'], marker='o')
plt.title('Average Score vs temperature')
plt.xlabel('temperature')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('avg_score_vs_temperature.png')

plt.figure()
plt.plot(grouped['temperature'], grouped['avg_ad_subtlety_score'], marker='o')
plt.title('Average Ad Subtlety Score vs temperature')
plt.xlabel('temperature')
plt.ylabel('Average Ad Subtlety Score')
plt.tight_layout()
plt.savefig('avg_subtle_score_vs_temperature.png')

plt.figure()
plt.plot(grouped['temperature'], grouped['product_name_presence_rate'], marker='o')
plt.title('Product Name Presence Rate vs temperature')
plt.xlabel('temperature')
plt.ylabel('Presence Rate (fraction)')
plt.tight_layout()
plt.savefig('avg_product_rate_vs_temperature.png')
