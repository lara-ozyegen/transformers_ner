import os
import json
import pandas as pd

# Define the model names and dataset names
model_names = ['scibert-ft', 'bluebert-ft', 'bertclinical-ft', 'bioclinicalbert-ft', 'deberta-ft']
dataset_names = ['mtsamples2', 'doc-patient', 'phee']

# Initialize an empty list to store the results
results = []

# Iterate over each model and dataset to read the respective JSON files
for model_name in model_names:
    for dataset_name in dataset_names:
        file_path = f'analysis/{model_name}/{dataset_name}/reports/fold0/binary_classification_report.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    report = json.load(file)
                    # Extract the macro average values for precision, recall, and f1-score
                    macro_avg = {
                        'Model': model_name,
                        'Dataset': dataset_name,
                        'Macro Precision': report['macro avg']['precision'],
                        'Macro Recall': report['macro avg']['recall'],
                        'Macro F1-Score': report['macro avg']['f1-score']
                    }
                    results.append(macro_avg)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")
        else:
            print(f"File not found: {file_path}")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Display the results DataFrame
print(results_df)
