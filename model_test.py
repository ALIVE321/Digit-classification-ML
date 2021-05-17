import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np

label_list_path = "./data/test_label.pickle"

def read_pickle(path):
    with open(path, "rb") as f:
        read_data = pickle.load(f)
    return np.array(read_data)

def prediction_report(pred_array, label_array):
    all_labels = list(range(10))
    print(f"Accuracy    : {accuracy_score(label_array, pred_array) * 100 :.3f}%")
    print(f"Precision   : {precision_score(label_array, pred_array, average='macro', labels=all_labels, zero_division=0) * 100 :.3f}%")
    print(f"Recall      : {recall_score(label_array, pred_array, average='macro', labels=all_labels) * 100 :.3f}%")
    print(f"F1-score    : {f1_score(label_array, pred_array, average='macro', labels=all_labels, zero_division=0)}")

param_parser = argparse.ArgumentParser(description="Choose Input Prediction File.")
param_parser.add_argument("input", type=str, help="Input File With Pickle Format.")
args = param_parser.parse_args()

pred_list_path = args.input



