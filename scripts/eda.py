# %%
%load_ext autoreload
%autoreload 2
import torch
import pandas as pd
from asd_tools.models import MobileNetV2,ASDMobileFaceNet
from asd_tools.utils import read_hdf5
import torchaudio.transforms as T
import torchvision.transforms.functional as F
import torchaudio
import matplotlib.pyplot as plt
import asd_tools.models
import yaml
# %%
t = torch.rand(32, 16000)
model = ASDMobileFaceNet()
y = model(t)
for k,v in y.items():
    print(k, v.shape)
# %%
df = pd.read_csv("/fsws1/i_kuroyanagi/asd_for_domain_shift/scripts/exp/bearing/disentangle.original_domain-1_seed0/best_loss/best_loss_agg.csv")
df
# %%
import keyword
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 1, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0

    # writer.add_embedding(
    #     embed,
    #     metadata=label,
    #     label_img=spec,
    #     global_step=self.epochs,
    #     tag="embedding",
    # )
writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), metadata=meta)
# %%
with open("conf/tuning/asd_model.002.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)
model_class = getattr(asd_tools.models, config["model_type"])
model = model_class(**config["model_params"])
state_dict = torch.load("exp/fan/asd_model.003/domain-1_seed0/checkpoint-3epochs/checkpoint-3epochs.pkl", map_location="cpu")
model.load_state_dict(state_dict["model"])
model.eval()
df = pd.read_csv("downloads/dev/bearing/blind_attributes_seed0.csv")
batch_embed = torch.empty((0, 128))
batch_spec = torch.empty((0, 1, 15, 28))
batch_path = []
for i in range(400):
    path = df.loc[i*10, "path"]
    title = "_".join(path.split("_")[1:5])
    batch_path.append(title)
    wave, sr = torchaudio.load(filepath=path)
    with torch.no_grad():
        y_ = model(wave, getspec=True)
    embed = y_["embedding"]
    spec = F.resize(y_["spec"], (15, 28))
    spec = (spec - spec.min()) / (spec.max() - spec.min())
    spec = (spec * 255).to(dtype=torch.uint8)
    print(i, spec.shape, embed.shape)
    batch_embed = torch.cat([batch_embed, embed])
    batch_spec = torch.cat([batch_spec, spec])
# %%
plt.figure()
plt.imshow(batch_spec[0].numpy().transpose(1,2,0))
plt.figure()
plt.hist(batch_spec[0].numpy().flatten())
writer = SummaryWriter()
writer.add_embedding(batch_embed,
                    metadata=batch_path,
                    label_img=batch_spec)
writer.close()
# %%
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
def select_n_random(data, labels, n=1000):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer = SummaryWriter()
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
# %%
df = pd.read_csv("exp/fan/asd_model.000/domain-1_seed0/best_loss/best_loss.csv")
df["path", "domain", "phase", "e1"].groupby(by="path").mean()
# %%
from asd_tools.models import MobileNetV2Extractor
import torch
model = MobileNetV2Extractor(width_mult=1)


y = model(torch.rand(1,3,128,64))
# %%
import torch
from asd_tools.models import VICReg
import torch.nn.functional as F
B = 32
section = F.softmax(torch.rand(B,6), dim=1)
print("section",section)
model = VICReg()
y_ = model(torch.rand(B,16000*2), section=section)


# %%
lam = torch.rand(16,1,1,1) * torch.rand(16,1,128,64)
# %%
