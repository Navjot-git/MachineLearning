# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
    'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total', 'spam'
]

df = pd.read_csv('spambase.data', header=None, names=column_names)

# Example: Check class balance (spam vs ham)
sns.countplot(x='spam', data=df)
plt.title('Distribution of Spam vs Ham Emails')
plt.show()

# Example: Correlation matrix heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize the distribution of some key features
plt.figure(figsize=(12, 6))

# Example: Distribution of 'word_freq_make'
sns.histplot(df['word_freq_make'], kde=True)
plt.title('Distribution of Word Frequency: "make"')
plt.show()

# Example: Distribution of 'char_freq_$'
sns.histplot(df['char_freq_$'], kde=True)
plt.title('Distribution of Character Frequency: "$"')
plt.show()

# Boxplot to detect outliers in word frequencies
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :-1])  # Exclude the target variable 'spam'
plt.xticks(rotation=90)
plt.title('Boxplot of Word and Character Frequencies')
plt.show()

# Pairplot for a few selected features and the target
sns.pairplot(df[['word_freq_make', 'char_freq_$', 'capital_run_length_average', 'spam']], hue='spam')
plt.show()

# Visualizing the difference in capital_run_length_average for spam and non-spam emails
plt.figure(figsize=(10, 6))
sns.boxplot(x='spam', y='capital_run_length_average', data=df)
plt.title('Capital Run Length Average for Spam vs Ham Emails')
plt.show()

