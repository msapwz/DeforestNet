import os
from torch.utils.data import DataLoader,Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import torch
from utils import get_loaders
# import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

def get_files(file,split="train"):
    
    file = os.path.join(file,split+".txt")
    if not os.path.exists(file):
        raise FileNotFoundError(f"The provided file do not exists: {file}")
    
    with open(file) as f:

        list_of_files = [i.replace("\n","") for i in f]
        return list_of_files

class ChangeDetectionDataset(Dataset):

    def __init__(self,
                root_dir,
                size,
                split="train",
                transform=None # for the
                ):

        self.root_dir = root_dir
        self.size = size
        self.i1_dir = os.path.join(root_dir,"A")
        self.i2_dir = os.path.join(root_dir,"B")
        self.lb_dir = os.path.join(root_dir,"label")
        self.list = os.path.join(root_dir,"list")

        self.files = get_files(self.list,split)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        name = self.files[index]
        
        A_ = os.path.join(self.i1_dir,name)
        B_ = os.path.join(self.i2_dir,name)
        label_ = os.path.join(self.lb_dir,name)


        A = read_image(A_)
        B = read_image(B_)
        label = read_image(label_)

        label = label.long()
        label = label // 255

        A = F.convert_image_dtype(A, dtype=torch.float)
        B = F.convert_image_dtype(B, dtype=torch.float)

        A = TF.normalize(A, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        B = TF.normalize(B, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        label = label.long()

        files = {
            "name":name,
            "A":A,
            "B":B,
            "L":label
        }
        
        return files
        


class Dataloaders():

    def __init__(self,args):

        root_dir = args.root_dir_dataset
        size = args.img_size
        batch = args.batch_size

        print("BATCH:",batch)
        if args.levir_loader=="True":
        # if root_dir.endswith("LEVIR"):
            print("USING LEVIR LOADER")
            self.train,self.val,self.test = get_loaders(args,root_dir)
        else:
            print("USING CHANGEDETECTION LOADER (noaug)")
            dataset_train = ChangeDetectionDataset(root_dir,size,"train")
            dataset_valid = ChangeDetectionDataset(root_dir,size,"val")
            dataset_test = ChangeDetectionDataset(root_dir,size,"test")

            dataloader_training = DataLoader(dataset=dataset_train,batch_size=batch,shuffle=True)
            dataloader_validation = DataLoader(dataset=dataset_valid,batch_size=batch,shuffle=True)
            dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch,shuffle=True)

            self.train = dataloader_training
            self.val = dataloader_validation
            self.test = dataloader_test




if __name__=="__main__":
    root_dir = "../datasets/LEVIR" 
    size = 256
    split = "train" 
    
    BATCH_SIZE = 10

    dataset_train = ChangeDetectionDataset(root_dir,size,"train")
    dataset_valid = ChangeDetectionDataset(root_dir,size,"val")

    dataloader_training     = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True)
    dataloader_validation   = DataLoader(dataset=dataset_valid,batch_size=BATCH_SIZE,shuffle=True)

    while True:
        files = next(iter(dataloader_training))
        x = files["A"]
        y = files["L"]
        if x.min()!=0 or x.max()!=1:
            print(f'x = min: {x.min()}; max: {x.max()}')
            break

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')