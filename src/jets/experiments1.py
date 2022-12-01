# %%


# %%
# Progress bar
import torch.optim as optim
from encodermodels import SimpleConvNetEncoder
from datasets import DataManager
from tqdm.notebook import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# %%
# Fetching the device that will be used throughout this notebook
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

# %%
# DEFINE DATA AND HYPERPARAMETERS
data_config = "QCD1"
tran_batch_size = 64
# %%
# DEFINE A DATASET MANAGER
DM = DataManager(data_config=data_config, transform=None)

# %%
train_loader = data.DataLoader(
    DM.train_set, batch_size=tran_batch_size, shuffle=False, drop_last=False)
val_loader = data.DataLoader(
    DM.val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(
    DM.test_bg, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

# %%


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=1.):
        super().__init__()

        self.logit_scale = logit_scale

        self.__loss_evol = {'train': [], 'valid': []}

    @property
    def loss_evolution(self):
        return self.__loss_evol

    def item(self):
        return self.item_

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device

        logits_1 = self.logit_scale * embedding_1 @ embedding_2.T
        logits_2 = self.logit_scale * embedding_2 @ embedding_1.T

        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) +
            F.cross_entropy(logits_2, labels)
        )

        self.__loss_evol['valid' if valid else 'train'].append(loss.item())
        self.item_ = loss.item()
        return loss


class ExpLoss(nn.Module):
    def __init__(self, logit_scale=1.):
        super().__init__()

    def item(self):
        return self.item_

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device

        loss = ((embedding_1-embedding_2)**2).mean()
        self.item_ = loss.item()
        return loss


# %%

clip_loss = CLIPLoss()
encoder1 = SimpleConvNetEncoder()
encoder1.to(device)
encoder2 = SimpleConvNetEncoder()
encoder2.to(device)
optimizer = optim.Adam(list(encoder1.parameters()) +
                       list(encoder2.parameters()), lr=0.0001)

# %%
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs = data[0].reshape(
            data[0].shape[0], data[0].shape[3], data[0].shape[1], data[0].shape[2]).float()
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        embedding1 = encoder1(inputs)
        embedding2 = encoder2(inputs)
        loss = clip_loss.forward(embedding1, embedding2)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += clip_loss.item()
        print_T = 20
        if i % print_T == print_T-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_T:.6f}')
            running_loss = 0.0


# %%
