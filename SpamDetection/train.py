import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Avoid line wrapping

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

# Load the dataset
df = pd.read_csv('spambase.data', header=None, names=column_names)

# Check the first few rows to see what the data looks like
# print(df.head())

# Check for missing values
print(df.isnull().sum())
# Check the shape of the dataset
print("Dataset shape:", df.shape)

# Check for missing values
print("Missing values:", df.isnull().sum().sum())

# Check class distribution (spam vs ham)
print(df['spam'].value_counts())

# Basic statistics for numeric features
#print(df.describe())

# Separate the features (X) and the target variable (y)
X = df.drop(columns=['spam'])  # Features (all columns except 'spam')
y = df['spam']  # Target (spam or not)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and test sets
print(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid (starting with a few key parameters)
param_dist = {
    'n_estimators': [100, 200, 300, 400],   # Number of trees
    'max_depth': [3, 5, 7],                 # Maximum depth of trees
    'learning_rate': [0.01, 0.1, 0.2],      # Step size for updates
    'subsample': [0.8, 1.0],                # Fraction of training data used
    'colsample_bytree': [0.8, 1.0]          # Fraction of features used
}


# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Perform random search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, 
                                   n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Fit the random search model
random_search.fit(X_train_scaled, y_train)

# Print the best parameters and best score (accuracy)
print("Best parameters found: ", random_search.best_params_)
print("Best accuracy found: ", random_search.best_score_)

# Train the XGBoost model
xgb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate XGBoost model
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
