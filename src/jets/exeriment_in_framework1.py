#from config_utils import Config
from open_clip import modified_resnet
import pickle
import matplotlib.pyplot as plt
import config_utils
import torch

config_file_path = "src/jets/config/default.yaml"
config = config_utils.Config(config_file_path)
cfg = config.get_dotmap()
print(cfg.config_file_path+cfg.QCD_images)

with open(cfg.jet_data_path+cfg.QCD_images, 'rb') as file:
    bg_im = pickle.load(file)
plt.imshow(bg_im[0])
plt.savefig("plots/one_image.png")


MRN = modified_resnet.ModifiedResNet(
    layers=[3, 4, 6, 3], output_dim=1024, heads=32, image_size=224, width=64)


y = MRN.forward(torch.tensor(bg_im[0].reshape(1, 1, 40, 40)).float())
