import pandas as pd
import os
import glob
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

answer_key = pd.read_csv('Day3-Activity-Results/student_code/answer_key.csv')

# Get list of all student submission files
submission_files = glob.glob('Day3-Activity-Results/student_code/*.csv')

# Remove answer key from list of files to grade
submission_files = [f for f in submission_files if 'answer_key.csv' not in f]

results = []
financial_predictions = {}

# Process each submission
for submission_file in submission_files:
    try:
        # Read student submission
        submission = pd.read_csv(submission_file)
        
        # Get student name from filename
        student = os.path.basename(submission_file).split('_')[0]
        
        # Ensure submission has IsFraud column
        if 'IsFraud' not in submission.columns:
            print(f"Error: {student}'s submission missing IsFraud column")
            continue
            
        # Merge submission with answer key on TransactionID if present, otherwise on index
        if 'TransactionID' in submission.columns:
            merged = pd.merge(answer_key[['TransactionID', 'IsFraud', 'Amount', 'TransactionDate']], 
                            submission[['TransactionID', 'IsFraud']], 
                            on='TransactionID',
                            suffixes=('_true', '_pred'))
        else:
            # Assume same order as answer key if no TransactionID
            merged = pd.DataFrame({
                'IsFraud_true': answer_key['IsFraud'],
                'IsFraud_pred': submission['IsFraud'][:len(answer_key)],
                'Amount': answer_key['Amount'],
                'TransactionDate': answer_key['TransactionDate']
            })
        
        # Convert TransactionDate to datetime
        merged['TransactionDate'] = pd.to_datetime(merged['TransactionDate'])
        
        # Sort by date
        merged = merged.sort_values('TransactionDate')
        
        # Calculate cumulative financial impact over time
        merged['revenue'] = merged.apply(lambda row: row['Amount'] * 0.02 if row['IsFraud_pred'] == 0 and row['IsFraud_true'] == 0 else 0, axis=1)
        merged['loss'] = merged.apply(lambda row: row['Amount'] if row['IsFraud_pred'] == 0 and row['IsFraud_true'] == 1 else 0, axis=1)
        merged['cumulative_impact'] = (merged['revenue'] - merged['loss']).cumsum()
        
        # Store financial predictions for this student
        financial_predictions[student] = merged[['TransactionDate', 'cumulative_impact']]
        
        # Calculate overall metrics (as before)
        overall_accuracy = (merged['IsFraud_true'] == merged['IsFraud_pred']).mean()
        roc_auc = roc_auc_score(merged['IsFraud_true'], merged['IsFraud_pred'])
        tn, fp, fn, tp = confusion_matrix(merged['IsFraud_true'], merged['IsFraud_pred']).ravel()
        fraud_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
        net_financial_impact = merged['cumulative_impact'].iloc[-1]
        
        results.append({
            'student': student,
            'overall_accuracy': overall_accuracy,
            'roc_auc': roc_auc,
            'fraud_accuracy': fraud_accuracy,
            'net_financial_impact': net_financial_impact,
            'submission_file': submission_file
        })
        
    except Exception as e:
        print(f"Error processing {submission_file}: {str(e)}")
        
# Create results dataframe
results_df = pd.DataFrame(results)
if len(results_df) > 0:
    print("\nGrading Results:")
    pd.set_option('display.float_format', lambda x: f'{x:.20f}')
    print(results_df.sort_values('net_financial_impact', ascending=False))
    
    # Plot financial predictions for each student
    plt.figure(figsize=(12, 6))
    for student, data in financial_predictions.items():
        plt.plot(data['TransactionDate'], data['cumulative_impact'], label=student)
    
    plt.xlabel('Transaction Date')
    plt.ylabel('Cumulative Financial Impact ($)')
    plt.title('Financial Predictions Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('financial_predictions.png')
    plt.close()
    
    print("\nFinancial prediction graph saved as 'financial_predictions.png'")
else:
    print("No valid submissions found")
