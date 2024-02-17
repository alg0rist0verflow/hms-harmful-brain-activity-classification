import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from glob import glob

def parquet2numpy(df, test_df, CFG):

    # Define a function to process a single eeg_id
    def process_spec(spec_id, split="train"):
        spec_path = f"{CFG.BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
        spec = pd.read_parquet(spec_path)
        np.savetxt(f"{CFG.SPEC_DIR}/{split}_spectrograms/{spec_id}.txt", 
                spec.columns.values, delimiter=" ", fmt="%s") 
        spec = spec.fillna(0).values[:, 1:].T # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
        spec = spec.astype("float32")
        np.save(f"{CFG.SPEC_DIR}/{split}_spectrograms/{spec_id}.npy", spec)

    # Get unique spec_ids of train and valid data
    spec_ids = df["spectrogram_id"].unique()

    # Parallelize the processing using joblib for training data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "train")
        for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    # Get unique spec_ids of test data
    test_spec_ids = test_df["spectrogram_id"].unique()

    # Parallelize the processing using joblib for test data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "test")
        for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids))
    )

    data_label_consistency_test = []
    for f in glob(f'{CFG.SPEC_DIR}/train_spectrograms/*.txt'):
        data_label_consistency_test.append('_'.join(np.loadtxt(f,dtype=str)))
        
    _set1 = set(data_label_consistency_test)
    assert len(_set1)==1
    print(_set1)


def parquetEEG2NPY(df, test_df, CFG):
    # load eeg
    def process_eeg(eeg_id, split="train"):
        eeg_path = f"{CFG.BASE_PATH}/{split}_eegs/{eeg_id}.parquet"
        eeg = pd.read_parquet(eeg_path)
        np.savetxt(f"{CFG.SPEC_DIR}/{split}_eegs/{eeg_id}.txt", 
                eeg.columns.values, delimiter=" ", fmt="%s") 
        eeg = eeg.fillna(0).values
        eeg = eeg.astype("float32")
        filename = f"{CFG.SPEC_DIR}/{split}_eegs/{eeg_id}.npy"
        np.save(filename, eeg)

    # Get unique spec_ids of train and valid data
    eeg_ids = df["eeg_id"].unique()

    # Parallelize the processing using joblib for training data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_eeg)(eeg_id, "train")
        for eeg_id in tqdm(eeg_ids, total=len(eeg_ids))
    )

    # Get unique spec_ids of test data
    test_eeg_ids = test_df["eeg_id"].unique()

    # Parallelize the processing using joblib for test data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_eeg)(eeg_id, "test")
        for eeg_id in tqdm(test_eeg_ids, total=len(test_eeg_ids))
    )

    from glob import glob
    data_label_consistency_test = []
    for f in glob(f'{CFG.SPEC_DIR}/train_eegs/*.txt'):
        data_label_consistency_test.append('_'.join(np.loadtxt(f,dtype=str)))
        
    _set1 = set(data_label_consistency_test)
    assert len(_set1)==1
    print(_set1)