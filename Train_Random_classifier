import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# File paths
save_dir = r"C:\Users\user\Downloads\model.p"
data_path = r"C:\Users\user\Downloads\data.pickle"

# Load data
data_dict = pickle.load(open(data_path, "rb"))

# Check if data has inconsistent shapes
data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Find the maximum length of feature vectors
max_length = max(len(i) for i in data_list)

# Pad all feature vectors to the same length
data = np.array([np.pad(i, (0, max_length - len(i)), mode='constant') for i in data_list])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open(save_dir, 'wb') as f:
    pickle.dump({'model': model}, f)
