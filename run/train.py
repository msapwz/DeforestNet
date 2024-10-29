import torch
import torch.nn.functional as F

import os,json

from dataset.dataloaders import Dataloaders
from utils import get_optimizer, get_loss,get_scheduler
from model.metrics import GetMetricFunctions,ConfuseMatrixMeter
from model.visual import plot_learning_curves,plt_images
from model.models import DeforestationModel

from tqdm import tqdm
import pandas as pd

from utils import Logger

import time

class Trainer():
    def __init__(self, args):
        self.args = args
        self.dataloaders = Dataloaders(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeforestationModel(
                args,
                embed_dim=256,
                input_nc=3, 
                output_nc=2, 
                decoder_softmax=False,
                fusion = args.fusion,                       #"concat", attention, average, sum
                encoder =args.encoder,                      #"SegFormer",SegNext
                # att_type=args.attention_type,               #"att",cta
                use_cta=args.use_cta,
                siamese_type=args.siamese_type,             #"single",dual
                feature_exchange=args.feature_exchange,     #"N",simple,learnable
                ).to(self.device)
        
        if self.args.pretrain!="":
            self.net.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net.to(self.device)
            self.net.eval()
        
        self.optimizer = get_optimizer(args, self.net.parameters())
        self.multi_scale_infer = False
        #self.multi_scale_train = True  
        self.multi_scale_train = args.multi_scale_train == "True"
        self.multi_pred_weights = [0.5, 0.5, 0.5, 0.8, 1.0]
        self.weights = tuple(self.multi_pred_weights)

        if self.multi_scale_train:
            print("Running MultiScale Train")
        else:
            print("NOT USING MultiScale Train")

        self.loss = get_loss(args, self.dataloaders)
        if args.loss_weight==0:
            self.loss_with_weights = False
            print("RUNNING LOSS WITH NO WEIGHTS")
        else:
            if not (0 < args.loss_weight <= 1):
                raise ValueError("loss_weight must be a number between 0 and 1 (exclusive of 0 and inclusive of 1)")

            weight = 1 - args.loss_weight
            self.loss_weight = torch.tensor([weight, args.loss_weight], device=self.device)
            self.loss_with_weights = True
            print("RUNNING LOSS WITH WEIGHTS:",self.loss_weight)
            #self.loss_weight = torch.tensor([0.2, 0.8],device=self.device) 
        self.calculated_loss = None
        self.label_predicted = None

        self.best_val_acc = 0.0

        #self.metrics = GetMetricFunctions()
        self.running_metric = ConfuseMatrixMeter(2)

        self.output_folder = args.output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.scheduler = get_scheduler(
            self.optimizer,
            args.lr_scheduler,
            args.epochs
            )
        
        logger_path = os.path.join(self.output_folder, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        
        # Log file setup
        # self.loss_weight = torch.tensor([0.2, 0.8],device=self.device) 
        self.log_file = os.path.join(self.output_folder, 'training_log.json')
        self.logs = []

        self.metrics_CF = []
        self.epoch_metrics = []

        self.is_training = True

        print("ARGS",args)
        self.logger.write(str(args))


    # def save_log(self):
    #     with open(self.log_file, 'w') as f:
    #         json.dump(self.logs, f, indent=4)

    def save_model(self):
        model_path = os.path.join(self.output_folder, f'Best_model.pth')#_epoch_{self.epoch_id+1}.pth')
        torch.save(self.net.state_dict(), model_path)

    def _backward(self,output,gt,backward=True):
        if self.multi_scale_train:
            i         = 0
            temp_loss = 0.0
            for pred in output:
                if pred.size(2) != gt.size(2):
                    if self.loss_with_weights:
                        temp_loss = temp_loss + self.weights[i]*self.loss(pred,F.interpolate(gt.float(), size=pred.size(2), mode="nearest"),self.loss_weight)
                    else:
                        temp_loss = temp_loss + self.weights[i]*self.loss(pred,F.interpolate(gt.float(), size=pred.size(2), mode="nearest"))
                else:
                    if self.loss_with_weights:
                        temp_loss = temp_loss + self.weights[i]*self.loss(pred, gt,self.loss_weight)
                    else:
                        temp_loss = temp_loss + self.weights[i]*self.loss(pred, gt)
                i+=1
            self.loss_v = temp_loss
        else:
            if self.loss_with_weights:
                self.loss_v = self.loss(output[-1], gt,self.loss_weight)
            else :
                self.loss_v = self.loss(output[-1], gt)

        if backward:
            self.loss_v.backward()

    # def _backward(self,output,gt,backward=True):
    #     if self.multi_scale_train:
    #         i         = 0
    #         temp_loss = 0.0
    #         for pred in output:
    #             if pred.size(2) != gt.size(2):
    #                 temp_loss = temp_loss + \
    #                     self.weights[i]*self.loss(
    #                     pred,
    #                     F.interpolate(
    #                         gt.float(), 
    #                         size=pred.size(2), 
    #                         mode="nearest"
    #                 )
    #                 )
    #             else:
    #                 temp_loss = temp_loss + \
    #                             self.weights[i]*self.loss(pred, gt)
    #             i+=1
    #         self.loss_v = temp_loss
    #     else:
    #         self.loss_v = self.loss(output[-1], gt)

    #     if backward:
    #         self.loss_v.backward()

    def save_images(self, inputs_A, inputs_B, outputs, labels, epoch,n,mode="train"):
        output_dir = os.path.join(self.output_folder,'training_images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        outputname = os.path.join(output_dir, f'{mode}_epoch_{epoch+1}_{n}.png')
        plt_images(inputs_A, inputs_B, outputs, labels, outputname)

    def get_messages(self,loss,epoch):

        scores = self.running_metric.get_scores()
        message = f'{epoch}, MIOU: {scores["miou"]:.4f} IOU_1: {scores["iou_1"]:.4f} loss: {loss:.4f}'
        return message
    
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                (self.is_training, self.epoch_id, self.args.epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

        mode = "train" if self.is_training else "val"

        cf = {"epoch":self.epoch_id,
            "loss":self.loss_v.data.cpu().item(),
            "mode":mode}
        cf_data = {**cf, **scores}
        self.epoch_metrics.append(cf_data)

        if not self.is_training:
            if scores["mf1"]>self.best_val_acc:
                self.best_val_acc = scores["mf1"]
                message = f"BEST ACCURACY (mF1) UPDATED!!\nNew value={self.best_val_acc}"
                print(message)
                self.logger.write(message)
                self.save_model()

    def _clear_cache(self):
        self.running_metric.clear()
    
    def collect_metrics(self,epoch,loss,mode):

        scores = self.running_metric.get_scores()

        cf = {"epoch":epoch,"loss":loss,"mode":mode}
        cf_data = {**cf, **scores}
        self.metrics_CF.append(cf_data)

    def update_metric(self,output,labels):
        o1 = output[-1].argmax(dim=1).detach().cpu().numpy()
        l1 = labels.detach().cpu().numpy()

        self.running_metric.update_cm(pr=o1,gt=l1)

    def train(self):
        
        for epoch in range(self.args.epochs):
            self.epoch_id = epoch

            for param_group in self.optimizer.param_groups:
                print(f"Epoch {epoch+1}/{self.args.epochs} - Learning Rate: {param_group['lr']}")

            # TRAIN
            self._clear_cache()
            self.net.train()
            count = 0
            running_loss = 0
            total = len(self.dataloaders.train)
            train_loader = tqdm(enumerate(self.dataloaders.train, 0), total=total)
            n = 0
            self.is_training = True
            for self.batch_id, batch in train_loader:
                inputs_A = batch['A']
                inputs_B = batch['B']
                labels = batch['L']

                inputs_A = inputs_A.to(self.device)
                inputs_B = inputs_B.to(self.device)
                labels = labels.to(self.device)
                
                output = self.net(inputs_A, inputs_B) 
                

                self.optimizer.zero_grad()
                # loss = self.loss(output, labels)
                # running_loss += loss.data.cpu().item()
                # loss.backward()
                self._backward(output,labels)
                running_loss += self.loss_v.data.cpu().item()
                self.optimizer.step()
                self.update_metric(output,labels)
                count+=1
                n+=1
                if  (count+1) % 10 == 0:

                    l = running_loss/10
                    
                    message = self.get_messages(l,epoch+1)
                    train_loader.set_description(message)

                    if  (count+1) % 100 == 0:
                        self.save_images(inputs_A, inputs_B, output[-1], labels, epoch,count)
                    self.collect_metrics(epoch,l,"train")
                    running_loss = 0

            self._collect_epoch_states()


            # EVAL
            self._clear_cache()
            self.net.eval()
            running_loss = 0
            count = 0
            total = len(self.dataloaders.val)
            val_loader = tqdm(enumerate(self.dataloaders.val,0),total=total)
            self.is_training = False
            with torch.no_grad():
                for self.batch_id, batch in val_loader:
                    inputs_A = batch['A']
                    inputs_B = batch['B']
                    labels = batch['L']

                    inputs_A = inputs_A.to(self.device)
                    inputs_B = inputs_B.to(self.device)
                    labels = labels.to(self.device)

                    output = self.net(inputs_A, inputs_B)  
                    self._backward(output,labels,backward=False)
                    running_loss += self.loss_v.data.cpu().item()
                    # loss = self.loss(output, labels)
                    # running_loss += loss.data.cpu().item()
                    
                    # loss.item()
                    self.update_metric(output,labels)
                    count+=1
                    if (count+1) % 10 == 0:
                        
                        #self.m = self.metrics.get_metrics(output[-1],labels)

                        l = running_loss/10
                        message = self.get_messages(l,epoch+1)
                        val_loader.set_description(message)

                        if  (count+1) % 100 == 0:
                            self.save_images(inputs_A, inputs_B, output[-1], labels, epoch,count,mode="val")
                        self.collect_metrics(epoch,l,"val")
                        running_loss = 0

            self._collect_epoch_states()

            # self.save_model(epoch)
            self.scheduler.step()
            

        df = pd.DataFrame(self.metrics_CF)
        df.to_csv(os.path.join(self.output_folder,"metrics.csv"))

        df_cf = pd.DataFrame(self.epoch_metrics)
        df_cf.to_csv(os.path.join(self.output_folder,"metrics_per_epoch.csv"))

        plot_learning_curves(df_cf,self.output_folder)
        print("Finished Training :)")