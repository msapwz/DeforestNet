from argparse import ArgumentParser
import utils
import os
from run.train import Trainer

def run():
    parser = ArgumentParser()
    parser.add_argument("--mode",default="train",help="Mode to run (train or eval)")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--root_dir_dataset',default="../datasets/LEVIR",type=str,help="Root directory, where the datasets are stored")
    parser.add_argument('--output_folder',default="outputs/LEVIR",type=str,help="Folder where outputs will be saved")
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--multi_scale_train', default="True", type=str)
    parser.add_argument('--levir_loader', default="True", type=str)
    parser.add_argument("--loss_weight",default=0,type=float,help="weight for class 1 (change), value between 0 and 1")

    parser.add_argument('--pretrain', default="", type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)

    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument("--fusion",default="concat",type=str,help="Type of fusion method, options are 'sum', 'average','concat' and 'attention' ")
    parser.add_argument("--encoder",default="SegFormer",type=str,help="Encoder type, options: 'SegFormer' and 'SegNext'")
    # parser.add_argument("--attention_type",default="att",type=str,help=' "att" or "cta" ')
    parser.add_argument("--use_cta",default="False",type=str)
    parser.add_argument("--siamese_type",default="single",type=str,help='"single" or "dual"')
    parser.add_argument("--feature_exchange",default="N,N,N,N",type=str,help='same number of Number of stages, N=None, se=SpatialExchange, ce=ChannelExchange,lc=LearnableChannel,ls=LearnableSpatial,lc2=LearnableChannel2,ls2=LearnableSpatial2')
    parser.add_argument("--init_spatial_scale",default=10.0,type=float,help='')
    parser.add_argument("--init_spatial_threshold",default=0.5,type=float,help='')
    parser.add_argument("--init_channel_threshold",default=0.5,type=float,help='')

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_scheduler', default='linear', type=str,
                        help='linear | step')
    
    parser.add_argument("--mscan_v",default="MSCAN_B",)

    args = parser.parse_args()
    utils.get_device(args)
    
    model = Trainer(args=args)
    model.train()


if __name__=="__main__":
    run()