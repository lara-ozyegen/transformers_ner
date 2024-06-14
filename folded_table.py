import os
import json
import pandas as pd
import numpy as np

# Define the model names and dataset names
model_names = ['biobert-ft', 'deberta-ft', 'scibert-ft', 'bertclinical-ft', 'bioclinicalbert-ft']
dataset_names = ['phee']

# Initialize an empty list to store the results
results = []

# Iterate over each model and dataset to read the respective JSON files
for model_name in model_names:
    for dataset_name in dataset_names:
        precisions = []
        recalls = []
        f1_scores = []
        for fold in range(5):
            file_path = f'analysis/{model_name}/{dataset_name}/reports/fold{fold}/multiclass_classification_report.json'
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    try:
                        report = json.load(file)
                        precisions.append(report['precision']['macro_wo_O'])
                        recalls.append(report['recall']['macro_wo_O'])
                        f1_scores.append(report['f1-score']['macro_wo_O'])
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        if precisions and recalls and f1_scores:
            # Calculate mean and std for each metric
            precision_mean = np.mean(precisions)
            precision_std = np.std(precisions)
            recall_mean = np.mean(recalls)
            recall_std = np.std(recalls)
            f1_mean = np.mean(f1_scores)
            f1_std = np.std(f1_scores)
            
            # Store the results in the list
            results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Macro Precision': f"{precision_mean:.4f} ± {precision_std:.4f}",
                'Macro Recall': f"{recall_mean:.4f} ± {recall_std:.4f}",
                'Macro F1-Score': f"{f1_mean:.4f} ± {f1_std:.4f}"
            })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
