#from config_utils import Config
from datasets import DataManager
from open_clip import modified_resnet
import pickle
import matplotlib.pyplot as plt
import config_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from encodermodels import SimpleConvNetEncoder

config_file_path = "src/jets/config/default.yaml"
config = config_utils.Config(config_file_path)
cfg = config.get_dotmap()
print(cfg.config_file_path+cfg.QCD_images)

# %%
DM = DataManager(train_data_name=train_data_name,
                 eval_data_name=eval_data_name, transform=transform)


# %% Create data loaders
train_loader = data.DataLoader(
    DM.train_set, batch_size=256, shuffle=False, drop_last=False)
val_loader = data.DataLoader(
    DM.val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(
    DM.test_bg, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

# %% Define CLIP loss


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


# %%Build all components of the model
clip_loss = CLIPLoss()
encoder1 = SimpleConvNetEncoder()
encoder2 = SimpleConvNetEncoder()
optimizer = optim.SGD(list(encoder1.parameters()) +
                      list(encoder2.parameters()), lr=0.001, momentum=0.9)
train_input = torch.Value(torch.from_numpy(bg_im), requires_grad=False)

# %% Start training
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for b in range(0, train_input.size(0), mini_batch_size)
    # get the inputs; data is a list of [inputs, labels]
    inputs = torch.tensor(data)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    embedding1 = encoder1(inputs)
    embedding2 = encoder2(inputs)
    loss = clip_loss(embedding1, embedding2, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += clip_loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')
