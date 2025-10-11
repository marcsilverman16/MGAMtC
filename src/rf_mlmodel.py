import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import openpyxl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import GridSearchCV 
import warnings 
import sys 
import os 
from rdkit import RDLogger
import random
from sklearn.metrics import precision_score, make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict
from captum.attr import FeaturePermutation
import torch


RDLogger.DisableLog('rdApp.*') #disable all RDKit warnings

os.makedirs("../data/Results/", exist_ok=True)
os.makedirs("../Figures", exist_ok=True)
os.makedirs("../data/Insights/", exist_ok=True)
# os.makedirs("../Figures/Model_Analysis", exist_ok=True)

class SklearnWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SklearnWrapper, self).__init__()
        self.model = model 
         
    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        probs = self.model.predict_proba(x_np)
        return torch.from_numpy(probs).float().to(x.device)
    
def captum_analysis(X, y, final_model, bits_to_keep):
    X_tensor = torch.FloatTensor(X) 
    wrapped_model = SklearnWrapper(final_model) 
    explainer = FeaturePermutation(wrapped_model) 
    attributions = explainer.attribute(X_tensor, target=1) 
    attributions_np = attributions.detach().cpu().numpy() 
    column_labels = [f'bit_{bit_idx}' for bit_idx in bits_to_keep] 
    attributions_df = pd.DataFrame(attributions_np, columns=column_labels) 
    attributions_df['label'] = y

    os.makedirs("../data/Results/Insights", exist_ok=True)
    attributions_df.to_excel('../data/Insights/feature_attributions_all.xlsx', index=False)

def plot_single_pr_curve(y_true, y_proba, fig_folder, model_name): 
    precision, recall, _ = precision_recall_curve(y_true, y_proba) 
    pr_auc = auc(recall, precision) 
    baseline = np.sum(y_true) / len(y_true) 
    plt.figure(figsize=(10, 8)) 
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})') 
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2, label=f'Random Classifier (AUC = {baseline:.3f})') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('Recall', fontweight='bold') 
    plt.ylabel('Precision', fontweight='bold') 
    plt.title(f'{model_name} Precision-Recall Curve', fontweight='bold', fontsize=14) 
    plt.legend(loc="lower left") 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() 
    plt.savefig(f"{fig_folder}/precision_recall_curve.svg", dpi=600, bbox_inches='tight') 
    plt.close()

def plot_probability_distribution(y_true, y_proba, fig_folder):
    plt.figure(figsize=(12, 6))
    
    proba_positive = y_proba[y_true == 1]
    proba_negative = y_proba[y_true == 0]
    
    bins = np.linspace(0, 1, 30)
    plt.hist(proba_negative, bins=bins, alpha=0.6, label='Non-binders (True Negative)', 
             color='steelblue', edgecolor='black', linewidth=0.5)
    plt.hist(proba_positive, bins=bins, alpha=0.6, label='Binders (True Positive)', 
             color='coral', edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Predicted Probability', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of Predicted Probabilities by True Class', fontweight='bold', fontsize=14)
    plt.legend(loc='upper center')
    plt.grid(axis='y', alpha=0.3)
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/probability_distribution.svg", dpi=600, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true, y_proba, fig_folder, n_bins=10):
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, 
             label='RandomForest', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='navy', 
             label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Fraction of Positives', fontweight='bold')
    plt.title('Calibration Curve (Reliability Diagram)', fontweight='bold', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/calibration_curve.svg", dpi=600, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y, y_pred, name): 
    fig_folder = "../Figures/Model_Analysis" 
    os.makedirs(fig_folder, exist_ok=True) 
    cm = confusion_matrix(y, y_pred) 
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
    plt.title('Confusion Matrix') 
    plt.ylabel('Actual') 
    plt.xlabel('Predicted') 
    plt.tight_layout() 
    plt.savefig(f"../Figures/Model_Analysis/{name}.svg", dpi=600, bbox_inches='tight')
    plt.close()

def plot_single_roc_curve(y_true, y_proba, fig_folder, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba) 
    roc_auc = auc(fpr, tpr) 
    plt.figure(figsize=(10, 8)) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate', fontweight='bold') 
    plt.ylabel('True Positive Rate', fontweight='bold') 
    plt.title(f'{model_name} ROC Curve', fontweight='bold', fontsize=14) 
    plt.legend(loc="lower right") 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() 
    plt.savefig(f"{fig_folder}/roc_curve.svg", dpi=600, bbox_inches='tight') 
    plt.close()

def plot_single_pr_curve(y_true, y_proba, fig_folder, model_name): 
    precision, recall, _ = precision_recall_curve(y_true, y_proba) 
    pr_auc = auc(recall, precision) 
    baseline = np.sum(y_true) / len(y_true) 
    plt.figure(figsize=(10, 8)) 
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})') 
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2, label=f'Random Classifier (AUC = {baseline:.3f})') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('Recall', fontweight='bold') 
    plt.ylabel('Precision', fontweight='bold') 
    plt.title(f'{model_name} Precision-Recall Curve', fontweight='bold', fontsize=14) 
    plt.legend(loc="lower left") 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() 
    plt.savefig(f"{fig_folder}/precision_recall_curve.svg", dpi=600, bbox_inches='tight') 
    plt.close()

def optimize_f1_threshold_cv(y_true, y_proba, threshold_range=(0.46, 0.56), step=0.02): 
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step) 
    results = [] 

    for threshold in thresholds: 
        y_pred = (y_proba >= threshold).astype(int) 
        f1 = f1_score(y_true, y_pred) 
        precision = precision_score(y_true, y_pred, zero_division=0) 
        recall = recall_score(y_true, y_pred, zero_division=0) 
        results.append({ 'threshold': threshold, 'f1_score': f1, 'precision': precision, 'recall': recall })

    results_df = pd.DataFrame(results) 
    best_row = results_df.loc[results_df['f1_score'].idxmax()] 
    optimal_threshold = best_row['threshold'] 
    print(f"\nOptimal threshold: {optimal_threshold:.2f}") 
    print(f"F1-score: {best_row['f1_score']:.2f}")
    print(f"Precision: {best_row['precision']:.2f}") 
    print(f"Recall: {best_row['recall']:.2f}") 

    return optimal_threshold

def createFingerprints(df): 
    fingerprints = [] 
    invalid_indices = [] #tracks indecies of invalid rows
    for idx, smiles in df["SMILES"].items(): 
        mol = Chem.MolFromSmiles(smiles) 
        if mol is not None: 
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) 
            fp_array = np.array(fp) 
            fingerprints.append(fp_array)            
        else: 
            invalid_indices.append(idx) 
            
    df = df.drop(index=invalid_indices).copy() #drop rows with invalid SMILES 
    df["Fingerprint"] = fingerprints 
    
    print(f"Dropped {len(invalid_indices)} invalid SMILES.") 
    return df

def filterFingerprints(df, pos_threshold=0.9, neg_threshold=0.95): 
    fp_length = len(df["Fingerprint"].iloc[0]) 
    
    #filter valid fingerprints for each class
    pos_fps = df[df["Label"] == 1] 
    neg_fps = df[df["Label"] == 0] 
    
    #stack the fingerprint arrays for mathematical operations 
    pos_fp_array = np.stack(pos_fps["Fingerprint"].values) 
    neg_fp_array = np.stack(neg_fps["Fingerprint"].values) 
    
    #calculate percentage of 1s for each bit position 
    pos_ones_percentage = np.mean(pos_fp_array == 1, axis=0) 
    neg_ones_percentage = np.mean(neg_fp_array == 1, axis=0) 
    
    #find conserved bits based on the thresholds 
    pos_conserved_0 = pos_ones_percentage <= (1 - pos_threshold) 
    pos_conserved_1 = pos_ones_percentage >= pos_threshold 
    neg_conserved_0 = neg_ones_percentage <= (1 - neg_threshold) 
    neg_conserved_1 = neg_ones_percentage >= neg_threshold 

    conserved_bits = (pos_conserved_0 & neg_conserved_0) | (pos_conserved_1 & neg_conserved_1) #considered a conserved bit location if conserved in both glass and bit has same value for both classes 
    
    bits_to_keep = np.where(~conserved_bits)[0] #gets rid of conserved bit locations
    print(f"Bits to keep: {len(bits_to_keep)} out of {fp_length}")
    # print(bits_to_keep)
    df_filtered = df.copy() 
    
    def filter_fingerprint(fp): 
        return fp[bits_to_keep].astype(int)
    
    df_filtered["Filtered_Fingerprint"] = df["Fingerprint"].apply(filter_fingerprint) 
    print("\nOriginal fingerprint length:", fp_length) 
    print("Number of bits removed:", fp_length - len(bits_to_keep)) 
    print("Filtered fingerprint length:", len(bits_to_keep)) 
    return df_filtered, bits_to_keep

def train_model(df, bits_to_keep): 
    X = df["Filtered_Fingerprint"] 
    y = df["Label"] 
    
    X = np.vstack(X) 
    
    #look at class imbalance 
    n_pos = sum(y == 1) 
    n_neg = sum(y == 0) 
    ratio = n_neg / n_pos 
    print("\nPositive class samples:", n_pos) 
    print("Negative class samples:", n_neg) 
    print(f"Class ratio (pos:neg): 1:{ratio:.2f}") 
    
    #given the little amount of data and class imabalnce, need to check a various combination of parameters 
    #this code can be changed depending on available computing power and amount of data input, and can test more combinations of parameters 
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 5, 8, 10, 15, None],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2], 
        "class_weight": ['balanced', {0:1, 1: ratio}, {0: 1, 1: 10}, {0: 1, 1: 20}] 
    } 
    
    seed_1 = random.randint(0, 10000) 
    print(f"\nRandom seed for parameter search: {seed_1}") 
    rf = RandomForestClassifier(random_state=seed_1) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_1) 
    
    #5 split Kfold cross-validation
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=params, 
        scoring='f1', 
        cv=cv, verbose=0, 
        n_jobs=-1 
    )

    grid_search.fit(X, y)

    print("\nBest parameters:", grid_search.best_params_)

    best_params = grid_search.best_params_

    seed_2 = random.randint(0, 10000) 
    print("\nRandom seed for cross-validation for unbias performance evaluation:", seed_2) 
    best_model_cv = RandomForestClassifier(**best_params, random_state=seed_2) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_2) 
    
    #cross-validated predictions (unbiased) 
    y_pred = cross_val_predict(best_model_cv, X, y, cv=cv) 
    y_proba = cross_val_predict(best_model_cv, X, y, cv=cv, method='predict_proba')[:, 1] 
    cv_results = {
        'test_f1': [f1_score(y, y_pred)], 
        'test_precision': [precision_score(y, y_pred)], 
        'test_recall': [recall_score(y, y_pred)], 
        'test_roc_auc': [roc_auc_score(y, y_proba)] 
    } 
    
    plot_single_roc_curve(y, y_proba, "../Figures/Model_Analysis", "RandomForest")
    plot_single_pr_curve(y, y_proba, "../Figures/Model_Analysis", "RandomForest") 
    plot_confusion_matrix(y, y_pred, "Original_Confusion_Matrix")
    plot_calibration_curve(y, y_proba, "../Figures/Model_Analysis")
    plot_probability_distribution(y, y_proba, "../Figures/Model_Analysis")

    #train final model on all data for threshold optimization & deployment 
    seed_3 = random.randint(0, 10000) 
    print("\nRandom seed for final model training:", seed_3) 
    final_model = RandomForestClassifier(**best_params, random_state=seed_3) 
    final_model.fit(X, y) 
    
    #captum feature analysis 
    captum_analysis(X, y, final_model, bits_to_keep)
    
    optimal_threshold = optimize_f1_threshold_cv(y, y_proba)
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int) 
    
    plot_confusion_matrix(y, y_pred_optimal, "Threshold_Optimized_Confusion_Matrix") 
    
    #for reporting, keep original df but add unbiased preds and probabilities 
    predictions_df = df.copy() 
    predictions_df['Probability'] = y_proba 
    predictions_df['Prediction'] = y_pred_optimal 
    performance_metrics = { 
        'classification_report': classification_report(y, y_pred_optimal, output_dict=True), 
        'optimal_threshold': optimal_threshold 
    }

    list_seeds = [seed_1, seed_2, seed_3]

    file_path = "../data/Meta/seeds.xlsx" 
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    df = pd.DataFrame([list_seeds], columns=["seed1", "seed2", "seed3"]) 
    df.to_excel(file_path, index=False) 
    return final_model, performance_metrics, predictions_df, optimal_threshold

def predict_new_data(model, new_data_df, bits_to_keep, optimal_threshold): 
    df_filtered = new_data_df.copy() 
    
    def filter_fingerprint(fp): 
        return fp[bits_to_keep].astype(int) 
    
    df_filtered["Filtered_Fingerprint"] = new_data_df["Fingerprint"].apply(filter_fingerprint) 
    
    X = df_filtered["Filtered_Fingerprint"] 
    X = np.vstack(X) 
    
    probabilities = model.predict_proba(X)[:, 1] 
    predictions = (probabilities >= optimal_threshold).astype(int) 
    
    results_df = new_data_df.copy() 
    results_df['Probability'] = probabilities 
    results_df['Prediction'] = predictions 
    
    pos_count = sum(predictions == 1) 
    neg_count = sum(predictions == 0) 
    print(f"Prediction results: {pos_count} positive, {neg_count} negative") 
    
    return results_df

#read data from an Excel file 
def read_data(file_classifications): 
    ext = os.path.splitext(file_classifications)[1].lower() 
    if ext in ['.xls', '.xlsx']: 
        df = pd.read_excel(file_classifications) 
    elif ext == '.csv': 
        df = pd.read_csv(file_classifications) 
    else: 
        print("File format not supported. Please provide an Excel or CSV file.") 
        
    return df

def to_excel(df): 
    sorted_df = df.sort_values(by="Probability", ascending=False) 
    output_path = "../data/Results/results_sorted.xlsx" 
    sorted_df.to_excel(output_path, index=False) 
    
def process_data(df): 
    df["Label"] = df.apply(lambda x : 1 if x["Binding"] == "Yes" else 0, axis=1) 
    fp = createFingerprints(df) 
    fp_filtered, bits_to_keep = filterFingerprints(fp) 
    return fp_filtered, bits_to_keep 

def main(): 
    file_classifications = sys.argv[1] #contains path for file that has experimentally tested classifications 
    file_screening = sys.argv[2] #contains path for file that has screening data 
    
    path = (file_classifications) 
    raw_data = read_data(path) 
    df, bits_to_keep = process_data(raw_data) 
    final_model, performance_metrics, predictions_df, optimal_threshold = train_model(df, bits_to_keep) 
    
    # print(df.head()) 
    
    raw_data_screening = read_data(file_screening) 
    fp_new_data = createFingerprints(raw_data_screening) 
    result_predictions = predict_new_data(final_model, fp_new_data, bits_to_keep, optimal_threshold) 
    to_excel(result_predictions)

if __name__ == "__main__":
    main()