import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load your mushroom dataset (update the file path)
data_path = "mushroom_training_data.csv"
df = pd.read_csv(data_path)

# Step 2: Initialize dictionary to hold encoders
label_encoders = {}

# Step 3: Fit LabelEncoders for all categorical features except target ('class')
for col in df.columns:
    if col == 'class':  # Skip target label
        continue
    le = LabelEncoder()
    le.fit(df[col])
    label_encoders[col] = le
    print(f"Fitted LabelEncoder for '{col}' with classes: {list(le.classes_)}")

# Step 4: Save the encoders dictionary to a pickle file
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("\nLabel encoders saved successfully as 'label_encoders.pkl'")
