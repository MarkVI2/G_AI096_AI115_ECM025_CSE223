import pandas as pd

def risk_score():
    """
    Computes a risk score based on classification and regression results.
    
    Returns:
        pd.DataFrame: DataFrame containing the risk scores.
    """
    # Load classification and regression results
    classification_results = pd.read_csv('classification_results.csv')
    regression_results = pd.read_csv('regression_results.csv')
    
    # Merge results on 'id' column
    merged_results = pd.merge(classification_results, regression_results, on='id')
    
    # Compute risk score
    merged_results['risk_score'] = (
        0.5 * merged_results['classification_score'] + 
        0.5 * merged_results['regression_score']
    )
    
    return merged_results[['id', 'risk_score']]
def main():
    """
    Main function to run the risk scoring process.
    """
    # Compute risk score
    risk_scores = risk_score()
    
    # Save results to CSV
    risk_scores.to_csv('risk_scores.csv', index=False)
    
    print("Risk scores computed and saved to 'risk_scores.csv'.")
if __name__ == "__main__":
    main()