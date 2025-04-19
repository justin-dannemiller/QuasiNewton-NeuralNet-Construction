###############################################################################
## Description: Define the function for loading the feature vectors and      ##
##              labels for the Breast Cancer Detection dataset from the      ##
##              University of Wisconsin.                                     ##
###############################################################################

import torch
from torch import Tensor
from typing import Tuple
import pandas as pd
import os

def prepare_wisc_breast_cancer_dataset(convert_to_tensor: bool = True) -> Tuple:
    """
        Description: Loads and preprocesses the University of Wisconsin
                     Breast Cancer Dataset
        Args:
            convert_to_tensor (bool): Converts X, y to tensors if true; otherwise
                                      X, y are each left as dataframes
        Returns:
            breast_cancer_dataset (X, y): Dataset containing the 699 samples, each having
                                           9 breast measurements as feature vectors (X) and
                                           the cancer diagnosis label (y)
    """
    dataset_path = os.path.join("Datasets", "Breast_Cancer_Dataset", "breast-cancer-wisconsin.data")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Provided path for dataset: {dataset_path} is not a valid path!")
    else:
        breast_cancer_df = pd.read_csv(dataset_path, header=None)

    # Set the column names of the dataframe
    column_names = [
        "Sample code number",
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "Class"
    ]
    breast_cancer_df.columns = column_names

    # Drop any input vectors with missing ('?') measurements
    breast_cancer_df.replace('?', pd.NA, inplace=True)
    breast_cancer_df.dropna(inplace=True)

    # Convert the data to numeric type and remap the class labels such that
    # malignant = 1, benign = 0
    for col in breast_cancer_df.columns[1:]:
        breast_cancer_df[col] = pd.to_numeric(breast_cancer_df[col])
    breast_cancer_df['Class'] = breast_cancer_df['Class'].map({4: 1, 2: 0})

    # Segment dataset into input vectors and labels
    X = breast_cancer_df.iloc[:, :-1]
    y = breast_cancer_df["Class"]
    if convert_to_tensor:
        X = X.drop(columns="Sample code number") # drop sample idx; keeping only feature vals
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
    breast_cancer_dataset = (X, y)
    return breast_cancer_dataset
