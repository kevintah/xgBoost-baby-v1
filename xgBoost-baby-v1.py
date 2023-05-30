# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read the CSV file into a DataFrame
df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Split the data into input features (x) and labels (y)
# and drop non-numeric values
x = df.drop(['Id', 'Class', 'EJ'], axis=1)
y = df['Class']

# Scale the data using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# Define the XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', random_state=0)

# Train the model
model.fit(x_train, y_train)

# Evaluate the model on the validation set
val_predictions = model.predict(x_val)
val_accuracy = np.mean(val_predictions == y_val)
print("Validation Accuracy:", val_accuracy)

# Perform KFold cross-validation
kf = KFold(n_splits=5, random_state=0, shuffle=True)

best_val_accuracy = 0
best_model = None
val_accuracies = []

for train_index, val_index in kf.split(x_scaled):
    x_train, x_val = x_scaled[train_index], x_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Define the XGBoost model
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=0)

    # Train the model
    model.fit(x_train, y_train)

    # Evaluate the model on the validation set
    val_predictions = model.predict(x_val)
    val_accuracy = np.mean(val_predictions == y_val)
    val_accuracies.append(val_accuracy)

    # Track the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model

# Train the final model with the best hyperparameters
model = best_model.fit(x_scaled, y)

# Plot the validation accuracies
sns.set()
plt.plot(range(1, kf.n_splits + 1), val_accuracies, '-o')
plt.xlabel('Fold')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy per Fold')
plt.show()

# Load the test data
test_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")

# Fill missing values with the mean
test_df.fillna(df.mean(), inplace=True)

# Scale the test data using StandardScaler
test_x_scaled = scaler.transform(test_df.drop(['Id', 'EJ'], axis=1))

# Predict the probabilities for the test data using the trained model
probabilities = model.predict_proba(test_x_scaled)

# Create a DataFrame for the predictions
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
sample['class_1'] = probabilities[:, 1]  # Probability for class 1
sample['class_0'] = probabilities[:, 0]  # Probability for class 0
sample.to_csv('submission.csv', index=False)
