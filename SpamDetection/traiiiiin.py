import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class SpamClassifier:
    
    def __init__(self, model_choice='xgboost'):
        # store the model choice
        self.model_choice = model_choice
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        # Load dataset
        column_names = column_names = [
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
        df = pd.read_csv(file_path, header=None, names=column_names)

        # Split into features and target
        X = df.drop(columns=['spam'])
        y = df['spam']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def preprocess_data(self):
        # Scale the data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def build_model(self):
        # Choose the model based on the flag
        if self.model_choice == 'svm':
            self.model = SVC(kernel='linear', random_state=42)
        elif self.model_choice == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        elif self.model_choice == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        elif self.model_choice == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)
        else:
            raise ValueError(f"Unknown model choice: {self.model_choice}")
        
    def train_and_evaluate(self):
        # Train the chosen model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Print evaluation metrics
        print(f"Evaluation for {self.model_choice.capitalize()}:")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
    def run(self, file_path):
        # Main workflow: load data, preprocess, build, train and evaluate
        self.load_data(file_path)
        self.preprocess_data()
        self.build_model()
        self.train_and_evaluate()
        
if __name__ == "__main__":
    # Create an instance of the classifier and choose a model by setting model_choice
    classifier = SpamClassifier(model_choice='xgboost') 
    classifier.run('spambase.data')