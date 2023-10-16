import numpy as np
import os
import sys
import torch
import pickle 
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class SepsisDataset(Dataset):
    def __init__(self, data_dir = None, is_train=True, window_size=12):
        self.data_dir = data_dir
        self.file_list = None
        self.is_train = is_train
        self.window_size = window_size
        self.train_files, self.test_files = None, None
        self.scaler = None

        self.windows = []
        self.targets = []
        
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]
    
    def setup(self):
        # Split the dataset into train and test sets
        self.file_list = [file for file in self.list_files_recursive(self.data_dir) if '.psv' in file]
        self.train_files, self.test_files = train_test_split(self.file_list, test_size=0.2, random_state=42)
        self.file_list = self.train_files if self.is_train else self.test_files

        for file in tqdm(self.file_list, desc=f"Creating rolling windows of size: {self.window_size} | Training: {self.is_train}"):
            window, target = self.process_psv_file(file, self.window_size)
            self.windows.extend(window)
            self.targets.extend(target.view(-1, 1))

        if self.is_train:
            # Initialize the StandardScaler for the training set
            self.scaler = StandardScaler()

            # Stack all windows and fit the scaler to the entire training set
            windows_stacked = torch.cat(self.windows, dim=0)
            self.scaler.fit(windows_stacked)

            # Transform the validation data using the fitted scaler
            self.windows = [torch.nan_to_num(torch.Tensor(self.scaler.transform(w)))
                            for w in tqdm(self.windows, desc="Scaling training data")]

    
    def list_files_recursive(self, root_dir):
        file_list = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        return file_list

    def process_psv_file(self, file_path, window_size=10):
        if '.psv' not in file_path:
            return None, None

        # Read the .psv file
        with open(file_path, 'r') as psv_file:
            lines = psv_file.readlines()
        variable_names = lines[0].strip().split('|')

        # Process the data for variable measurements
        time_series_data = []
        for line in lines[1:]:
            values = line.strip().split('|')
            values = [np.nan if value == 'nan' else float(value) for value in values]
            time_series_data.append(values)

        time_series_data = np.array(time_series_data, dtype=np.float32)

        # Handle case when window_size is longer than time_series_data
        if window_size > time_series_data.shape[0]:
            num_padding_rows = window_size - time_series_data.shape[0]
            padding_shape = (num_padding_rows, time_series_data.shape[1] - 1)
            padding = np.nan * np.ones(padding_shape)
            padding = np.column_stack((padding, np.repeat(time_series_data[0,-1], num_padding_rows)))
            time_series_data = np.concatenate((padding, time_series_data))

        # Perform linear interpolation for missing values in each column
        for col_index in range(time_series_data.shape[1]):
            col_values = time_series_data[:, col_index]
            indices = np.arange(len(col_values))
            mask = np.isnan(col_values)

            if np.any(mask):
                x_interp = indices[mask]
                y_interp = col_values[~mask]

                if y_interp.shape[0] == 1:
                    col_values[mask] = np.nan
                    continue

                elif y_interp.shape[0] == 0:
                    continue

                # Interpolate missing values
                interp_func = interp1d(indices[~mask], y_interp, kind='linear', fill_value='extrapolate')
                col_values[mask] = interp_func(x_interp)

        # Check if there are still NaNs after interpolation
        if np.isnan(col_values).any():
            # Replace remaining NaNs with the mean of each column
            mean = np.nanmean(col_values)
            nan_indices = np.isnan(col_values)
            col_values[nan_indices] = mean

        # Generate rolling windows
        windows = [torch.Tensor(time_series_data[i:i + window_size, :-1])
                   for i in range(time_series_data.shape[0] - window_size + 1)]

        target = time_series_data[window_size:, -1].copy()
        # Generate the target vector
        if target.shape[0] == 0:
            target = torch.Tensor([time_series_data[-1,-1]])
        else:
            target = torch.Tensor(np.concatenate((target, [target[-1]])))
        return windows, target
