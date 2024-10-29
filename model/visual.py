


import os
import random
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np
import cv2

def aggregate_by_epoch(df, metric):
    return df.groupby('epoch')[metric].mean()


def cvt_to_colored(ar,lb):
        ar = cv2.split(ar)[0]
        lb = cv2.split(lb)[0]
        
        out = np.zeros(lb.shape,dtype=int)
        out = np.where((lb == 0) & (ar == 0), 0,        # No deforestation
            np.where((lb == 255) & (ar == 255), 1,    # True Positive
            np.where((lb == 255) & (ar == 0), 2,      # False negative
            np.where((lb == 0) & (ar == 255), 3, out) # False Positive
            )))    
        return out

def plot_learning_curves(df,output_folder):

        df_train = df[df["mode"]=="train"]
        df_val = df[df["mode"]=="val"]

        df_train.drop(columns=["mode"],inplace=True)
        df_val.drop(columns=["mode"],inplace=True)

        ## IF NEEDED UNCOMMENT THIS
        # df_train = df_train.groupby("epoch").agg(lambda x: np.mean(x))
        # df_val = df_val.groupby("epoch").agg(lambda x: np.mean(x))

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(df_train.index, df_train['loss'], label='Train Loss', linestyle='-', marker='o')
        plt.plot(df_val.index, df_val['loss'], label='Validation Loss', linestyle='-', marker='o')
        # plt.plot(df_train['step'], df_train['loss'], label='Train Loss', linestyle='-', marker='o')
        # plt.plot(df_val['step'], df_val['loss'], label='Validation Loss', linestyle='-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot mIoU
        plt.subplot(1, 2, 2)
        # plt.plot(df_train['step'], df_train['miou'], label='Train mIoU', linestyle='-', marker='o')
        # plt.plot(df_val[  'step'], df_val['miou'], label='Validation mIoU', linestyle='-', marker='o')
        plt.plot(df_train.index, df_train['miou'], label='Train mIoU', linestyle='-', marker='o')
        plt.plot(df_val.index, df_val['miou'], label='Validation mIoU', linestyle='-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Training and Validation mIoU')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'learning_curves.png'))
        plt.close()

def cvt_img(img):
    # Scale from [-1, 1] to [0, 255]
    img = (img + 1.0) * 127.5
    # Convert from float32 to uint8
    img = img.astype(np.uint8)
    return img


def plt_images(imgs1, imgs2, outputs, labels, outputname):
    # Number of samples to display
    num_samples = min(4, len(imgs1))
    
    f, axarr = plt.subplots(num_samples, 5, figsize=(21, 21))

    cmap = plt.cm.colors.ListedColormap([
        'black',    # No deforestation
        'green',    # True Positive
        'red',      # False negative
        'yellow'    # False Positive
        ])
    
    for i in range(num_samples):
        img1 = imgs1[i].cpu().numpy().transpose((1, 2, 0))
        img2 = imgs2[i].cpu().numpy().transpose((1, 2, 0))
        label = labels[i].cpu().numpy().squeeze()
        output = outputs[i].argmax(dim=0).cpu().detach().numpy().squeeze()

        label[label==1]=255
        output[output==1]=255
        compare = cvt_to_colored(output,label)
        # output = outputs[i].cpu().detach().numpy().squeeze()
        # output = outputs[i].cpu().detach().numpy().squeeze()
        # output = outputs[i].argmax(dim=1).cpu().numpy().squeeze()
        
        axarr[i, 0].imshow(cvt_img(img1))
        axarr[i, 1].imshow(cvt_img(img2))
        axarr[i, 2].imshow(label, cmap='gray')
        axarr[i, 3].imshow(output, cmap='gray')
        axarr[i, 4].imshow(compare, cmap=cmap,interpolation='nearest')
        
        for j in range(5):
            axarr[i, j].tick_params(left=False, bottom=False)
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_xticklabels([])

    axarr[0, 0].set_title('Image 1')
    axarr[0, 1].set_title('Image 2')
    axarr[0, 2].set_title('Label')
    axarr[0, 3].set_title('Output')
    axarr[0, 4].set_title('Color compare')

    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputname)
    plt.close(f)




def visual_colored():
    # image1 = 'Tiny_model_4_CD/chipCF/NGB/256/A'
    # image2 = 'Tiny_model_4_CD/chipCF/NGB/256/B'
    # label = 'Tiny_model_4_CD/chipCF/NGB/256/label'
    # dataset_cf = 'ChangeFormer/best_cf/vis/CD_1679307466.36084153_ChangeFormerV6_Deforestation-256-NGB_b4_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/all'
    # dataset_oc = 'open-cd/infdir_deforestation/vis_data/vis_image'
    # dataset_tn = 'Tiny_model_4_CD/chipCF/NGB/256/compare_tiny_opencd'

    N = 0
    while True:
        output = f'comparison_tiny_changer_CF/colored/{N}.png'
        if not os.path.exists(output):
            break
        else:
            N+=1

    ds_cf = os.listdir(dataset_cf)
    ds_oc = os.listdir(dataset_oc)
    ds_tn = os.listdir(dataset_tn)

    assert len(ds_cf) == len(ds_oc) == len(ds_tn)

    images = 6

    random.shuffle(ds_cf)
    sample = random.sample(ds_cf, images)

    f, axarr = plt.subplots(6,6,figsize=(21,21))

    axarr[0, 0].set_title('Image 1')
    axarr[0, 1].set_title('Image 2')
    axarr[0, 2].set_title('Label')
    axarr[0, 3].set_title("\n".join(wrap("Change Former", 7)))
    axarr[0, 4].set_title('Open CD')
    axarr[0, 5].set_title('Tiny CD')

    cmap = plt.cm.colors.ListedColormap([
        'black',    # No deforestation
        'green',    # True Positive
        'red',      # False negative
        'yellow'    # False Positive
        ])

    import numpy as np

    def cvt_to_colored(ar,lb):
        ar = cv2.split(ar)[0]
        lb = cv2.split(lb)[0]
        
        out = np.zeros(lb.shape,dtype=int)
        out = np.where((lb == 0) & (ar == 0), 0,        # No deforestation
            np.where((lb == 255) & (ar == 255), 1,    # True Positive
            np.where((lb == 255) & (ar == 0), 2,      # False negative
            np.where((lb == 0) & (ar == 255), 3, out) # False Positive
            )))    
        return out


    for n,img in enumerate(sample):
        im1 = cv2.imread(os.path.join(image1,img))
        im2 = cv2.imread(os.path.join(image2,img))

        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        axarr[n,0].imshow(im1)
        axarr[n,1].imshow(im2)

        lb = cv2.imread(os.path.join(label,img))
        lb[lb==1]=255
        axarr[n,2].imshow(lb)

        cf_img = cv2.imread(os.path.join(dataset_cf,img))
        cf_img = cvt_to_colored(cf_img,lb)
        axarr[n,3].imshow(cf_img, cmap=cmap, interpolation='nearest')

        oc_img = cv2.imread(os.path.join(dataset_oc,img))
        oc_img = cvt_to_colored(oc_img,lb)
        axarr[n,4].imshow(oc_img, cmap=cmap, interpolation='nearest')

        tn_img = cv2.imread(os.path.join(dataset_tn,img))
        tn_img[tn_img==1]=255
        tn_img = cvt_to_colored(tn_img,lb)

        axarr[n,5].imshow(tn_img, cmap=cmap, interpolation='nearest')

        # plt.imshow(image_data, cmap=cmap, interpolation='nearest')
        # plt.colorbar()  # Add a colorbar for reference

        for i in range(6):

            axarr[n,i].tick_params(left = False,bottom=False) 
            axarr[n,i].set_yticklabels([])
            axarr[n,i].set_xticklabels([])
    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output)#,dpi=300)
    # plt.show()