import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from stamp.modeling.marugoto.transformer.data import get_cohort_df


def init_train_val_test(clini_table,
                        slide_table,
                        feature_dir,
                        target_label,
                        categories=["0", "1"],
                        random_state=42
                        ):
    categories = np.array(categories)
    target_enc = OneHotEncoder(sparse_output=False).fit(categories.reshape(-1, 1))

    df = get_cohort_df(clini_table, slide_table, feature_dir, target_label, categories)

    train_patients, test_patients = train_test_split(df.PATIENT,
                                                     test_size=0.2,
                                                     stratify=df[target_label],
                                                     random_state=random_state)
    train_df = df[df.PATIENT.isin(train_patients)]
    test_df = df[df.PATIENT.isin(test_patients)]

    train_patients, valid_patients = train_test_split(train_df.PATIENT,
                                                      stratify=train_df[target_label],
                                                      random_state=random_state)

    return train_df, test_df, valid_patients, target_enc
