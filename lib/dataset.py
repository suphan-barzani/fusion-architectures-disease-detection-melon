
import torch

import numpy as np

class BotryDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, class_):
        self.data = dataframe
        self.mean = self.data.iloc[:, 9:].mean(0)
        self.std = self.data.iloc[:, 9:].std(0)
        self.class_ = class_

        self.fluor_imgs = []

        for idx in range(dataframe.shape[0]):
            exp = self.data.iloc[idx]['exp']
            dpi = self.data.iloc[idx]['dpi']
            class_ = 'BC' if self.data.iloc[idx]['class'] == 'botrytis' else 'C'
            leaf = self.data.iloc[idx]['leaf']
            spot = self.data.iloc[idx]['spot']
            path = f'data/fluor/exp{exp}/{dpi}/{class_}{leaf}.tar-p{spot-1}.npy'

            img = np.load(path)
            fluor_data_square = torch.from_numpy(img).to(torch.float32)

            self.fluor_imgs.append(fluor_data_square)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        spectrum = (self.data.iloc[idx, 9:]-self.mean)/self.std
        spectrum = spectrum.values.astype(float)
        spectrum = torch.from_numpy(spectrum).to(torch.float32)

        fluor_data_line = self.data.iloc[idx, 5:9].values.astype(float)
        fluor_data_line = torch.from_numpy(fluor_data_line).to(torch.float32)

        # exp = self.data.iloc[idx]['exp']
        # dpi = self.data.iloc[idx]['dpi']
        # class_ = 'BC' if self.data.iloc[idx]['class'] == 'botrytis' else 'C'
        # leaf = self.data.iloc[idx]['leaf']
        # spot = self.data.iloc[idx]['spot']
        # path = f'data/fluor/exp{exp}/{dpi}/{class_}{leaf}.tar-p{spot-1}.npy'

        # img = np.load(path)
        # fluor_data_square = torch.from_numpy(img).to(torch.float32)

        fluor_data_square = self.fluor_imgs[idx]

        if self.class_ == 'dpi':
            target = 0.0 if self.data['dpi'].iloc[idx] == '1dpi' else 1.0
        elif self.class_ == 'group':
            target = 0.0 if self.data['class'].iloc[idx] == 'control' else 1.0

        target = torch.tensor(target).to(torch.float32)

        return (spectrum, fluor_data_line, fluor_data_square), target