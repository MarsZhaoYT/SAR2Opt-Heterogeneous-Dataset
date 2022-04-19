import torch
from torch.utils.data import Dataset

class AuxAttnDataset(Dataset):
    def __init__(self, size_A, size_B, device, init_method='one', mask_size=256, given_constant=1):
        super(AuxAttnDataset, self).__init__()
        self.A_size, self.B_size = size_A, size_B
        if init_method == 'one':
            self.A_attns = torch.ones((size_A, 1, mask_size, mask_size), dtype=torch.float32) * given_constant
            self.B_attns = torch.ones((size_B, 1, mask_size, mask_size), dtype=torch.float32) * given_constant
        elif init_method == 'random':
            self.A_attns = torch.rand((size_A, 1, mask_size, mask_size), dtype=torch.float32)
            self.B_attns = torch.rand((size_B, 1, mask_size, mask_size), dtype=torch.float32)

        self.A_attns = self.A_attns.to(device)
        self.B_attns = self.B_attns.to(device)

    def __getitem__(self, index_tuple):
        raise NotImplementedError('Auxiliary Dataset, doesn\'t support this operation')

    def __len__(self):
        return max(self.A_size, self.B_size)

    def get_attn_map(self, adx, bdx):
        return self.A_attns[adx], self.B_attns[bdx]

    def update_attn_map(self, idx, tgt_tensor, a_flag=True):
        if a_flag:
            self.A_attns[idx] = tgt_tensor
        else:
            self.B_attns[idx] = tgt_tensor