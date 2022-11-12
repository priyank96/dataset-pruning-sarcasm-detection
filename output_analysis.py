import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('model_results.csv')
    # only the wrongly classified test rows
    df = df[df['true_label'] != df['model_label']]
    df = df[['tweet', 'model_label', 'true_label']]
    df[df['true_label'] == 1].to_csv('false_negatives.csv', index=False)
    df[df['true_label'] == 0].to_csv('false_positives.csv', index=False)
