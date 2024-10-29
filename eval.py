import torch
from argparse import ArgumentParser
import os
import pandas as pd

from dataset.dataloaders import Dataloaders
from model.metrics import ConfuseMatrixMeter
from model.models import DeforestationModel
from model.visual import plt_images

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = Dataloaders(args)
    
    net = DeforestationModel(
        args,
        embed_dim=args.embed_dim,
        input_nc=3,  
        output_nc=args.n_class,
        decoder_softmax=False,
        fusion=args.fusion,
        encoder=args.encoder,
        use_cta=args.use_cta,
        siamese_type=args.siamese_type,
        feature_exchange=args.feature_exchange
    ).to(device)

    total_params = count_parameters(net)
    print(f"Total trainable parameters: {total_params}")

    if args.model_checkpoint != "":
        print(f"Loading model from {args.model_checkpoint}")
        net.load_state_dict(torch.load(args.model_checkpoint))
        net.to(device)
        net.eval()
    else:
        raise ValueError("You must provide a valid model checkpoint to run evaluation.")

    running_metric = ConfuseMatrixMeter(args.n_class)
    
    net.eval()
    total = len(dataloaders.test)
    val_loader = tqdm(enumerate(dataloaders.test, 0), total=total)

    with torch.no_grad():
        for batch_id, batch in val_loader:
            inputs_A = batch['A'].to(device)
            inputs_B = batch['B'].to(device)
            labels = batch['L'].to(device)

            output = net(inputs_A, inputs_B)

            o1 = output[-1].argmax(dim=1).detach().cpu().numpy()
            l1 = labels.detach().cpu().numpy()
            running_metric.update_cm(pr=o1, gt=l1)
            
            save_images(inputs_A, inputs_B, output[-1], labels, batch_id)

    scores = running_metric.get_scores()
    print("\nEvaluation Complete:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
    
    metrics_path = os.path.join(args.output_folder, "evaluation_metrics.csv")
    pd.DataFrame([scores]).to_csv(metrics_path)
    print(f"Metrics saved to {metrics_path}")


def save_images(inputs_A, inputs_B, outputs, labels, batch_id):
    output_dir = os.path.join(args.output_folder, 'eval_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outputname = os.path.join(output_dir, f'eval_batch_{batch_id}.png')
    plt_images(inputs_A, inputs_B, outputs, labels, outputname)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--root_dir_dataset', default="../datasets/LEVIR", type=str, help="Root directory where the datasets are stored")
    parser.add_argument('--output_folder', default="outputs/LEVIR", type=str, help="Folder where outputs will be saved")
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--levir_loader', default="True", type=str)

    parser.add_argument('--pretrain', default="", type=str)
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument("--fusion", default="concat", type=str, help="Type of fusion method, options are 'sum', 'average', 'concat', and 'attention'")
    parser.add_argument("--encoder", default="SegFormer", type=str, help="Encoder type, options: 'SegFormer' and 'SegNext'")
    parser.add_argument("--use_cta", default="False", type=str, help='')
    parser.add_argument("--siamese_type", default="single", type=str, help='"single" or "dual"')
    parser.add_argument("--feature_exchange", default="N,N,N,N", type=str, help='Same number of stages, options like N=None, se=SpatialExchange, ce=ChannelExchange')
    parser.add_argument("--mscan_v",default="MSCAN_B",)
    parser.add_argument("--init_spatial_scale",default=10.0,type=float,help='')
    parser.add_argument("--init_spatial_threshold",default=0.5,type=float,help='')
    parser.add_argument("--init_channel_threshold",default=0.5,type=float,help='')

    parser.add_argument('--model_checkpoint', type=str, default="outputs/LEVIR/Best_model.pth", help="Path to the trained model checkpoint")

    args = parser.parse_args()

    run_evaluation(args)
