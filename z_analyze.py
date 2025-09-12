import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame and 'numerical_column' is the column you want to plot
df = pd.read_csv('train_for_stats.csv')
# df['trip_duration_minutes'].hist(bins=2000) # Adjust 'bins' as needed

df = df[(df['trip_duration_minutes'] >= 0) & (df['trip_duration_minutes'] <= 120)].copy()

print(df['trip_duration_minutes'].std())
print(df['trip_duration_minutes'].mean())
print(df['trip_duration_minutes'].max())
print(df['trip_duration_minutes'].min())

# plt.title('Distribution of Numerical Column')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# # plt.show()

 # Assuming 'df' is your DataFrame and 'numerical_column' is the column you want to plot
# sns.kdeplot(df['trip_duration_minutes'])
# plt.title('KDE Plot of Numerical Column')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.show()

# sns.histplot(x = df['trip_duration_minutes'], color= "lightblue", edgecolor= None)
# plt.title("Distribution of trip in munites", pad = 15)
# plt.show()