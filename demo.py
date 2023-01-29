"""
ProHMR demo script.
To run our method you need a folder with images and corresponding OpenPose detections.
These are used to crop the images around the humans and optionally to fit the SMPL model on the detections.

Example usage:
python demo.py --checkpoint=path/to/checkpoint.pt --img_folder=/path/to/images --keypoint_folder=/path/to/json --out_folder=/path/to/output --run_fitting

Running the above will run inference for all images in /path/to/images with corresponding keypoint detections.
The rendered results will be saved to /path/to/output, with the suffix _regression.jpg for the regression (mode) and _fitting.jpg for the fitting.

Please keep in mind that we do not recommend to use `--full_frame` when the image resolution is above 2K because of known issues with the data term of SMPLify.
In these cases you can resize all images such that the maximum image dimension is at most 2K.
"""
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

# Signal handling
import signal
def print_signal(sig, frame):
    sig = signal.Signals(sig)
    print("Script recieved signal:", sig)
    if sig in [signal.SIGTERM, signal.SIGINT, signal.SIGCONT, signal.SIGUSR1]:
        print(f"{sig.name} recieved, raising SIGINT")
        raise InterruptedError(sig.name)

signal.signal(signal.SIGTERM, print_signal)
signal.signal(signal.SIGCONT, print_signal)
signal.signal(signal.SIGUSR1, print_signal)

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer

parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument("--img_name_filter", type=str, default='.*', help='Filter for image names. Only images with names matching this filter will be processed.')
parser.add_argument("--extension", type=str, default='jpg', help='Image extension to process.')
parser.add_argument("--render_viz", action='store_true', help='If set, render the visualization of the fitting')
parser.add_argument("--use_hips", action='store_true', help='If set, use hips in fitting')
parser.add_argument("--save_regression", action='store_true', help='If set, save regression results')


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

if args.run_fitting:
    keypoint_fitting = KeypointFitting(model_cfg)

def extra_filter(img_path, personid):
    # Run on image only if filter returns True
    img_fn, _ = os.path.splitext(os.path.relpath(img_path, args.img_folder))
    img_fn = f'{img_fn}_p{personid}'
    fit_path = os.path.join(args.out_folder, f'{img_fn}_fitting.npz')
    return not os.path.exists(fit_path)

# Create a dataset on-the-fly
dataset = OpenPoseDataset(model_cfg,
                        img_folder=args.img_folder,
                        keypoint_folder=args.keypoint_folder,
                        max_people_per_image=None,
                        img_name_filter=args.img_name_filter,
                        walk_subdirectories=True,
                        extra_filter=extra_filter,
                        extension=args.extension)

# Setup a dataloader with batch_size = 1 (Process images sequentially)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, drop_last=False)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# Go over each image in the dataset
for i, batch in enumerate(tqdm(dataloader)):

    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
        out_dict = {
            'global_orient': out['pred_smpl_params']['global_orient'].squeeze(2).detach().cpu().numpy(),
            'body_pose': out['pred_smpl_params']['body_pose'].detach().cpu().numpy(),
            'betas': out['pred_smpl_params']['betas'].detach().cpu().numpy(),
            'camera_translation': out['pred_cam_t'].detach().cpu().numpy(),
        }
    if args.run_fitting:
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=args.use_hips,
                                                full_frame=args.full_frame)
        opt_out_dict = {
            'global_orient': opt_out['smpl_params']['global_orient'].squeeze(1).detach().cpu().numpy(),
            'body_pose': opt_out['smpl_params']['body_pose'].detach().cpu().numpy(),
            'betas': opt_out['smpl_params']['betas'].detach().cpu().numpy(),
            'camera_translation': opt_out['camera_translation'].detach().cpu().numpy(),
            'losses': {k: v.detach().cpu().numpy() for k,v in opt_out['losses'].items()},
        }

    batch_size = batch['img'].shape[0]
    for n in range(batch_size):
        img_fn, _ = os.path.splitext(os.path.relpath(batch['imgname'][n], args.img_folder))
        personid = batch['personid'][n]
        img_fn = f'{img_fn}_p{personid}'

        # Make output directory since it may not exist
        Path(os.path.join(args.out_folder, img_fn)).parent.mkdir(parents=True, exist_ok=True)

        # Save result to disk
        if args.save_regression:
            np.savez(os.path.join(args.out_folder, f'{img_fn}_regression.npz'),
                                **{k:v[n] for k,v in out_dict.items()})
        if args.run_fitting:
            np.savez(os.path.join(args.out_folder, f'{img_fn}_fitting.npz'),
                     **{k:v[n] if not isinstance(v, dict) else {k2:v2[n] for k2,v2 in v.items()} for k,v in opt_out_dict.items()})

        if not args.render_viz:
            continue

        # Visualization
        regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                  out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                  batch['img'][n])
        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_regression.{args.out_format}'), 255*regression_img[:, :, ::-1])

        if args.run_fitting:
            fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                   opt_out['camera_translation'][n].detach().cpu().numpy(),
                                   batch['img'][n], imgname=batch['imgname'][n], full_frame=args.full_frame)
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_fitting.{args.out_format}'), 255*fitting_img[:, :, ::-1])
