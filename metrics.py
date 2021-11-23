import os
import torch

import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from torchvision.transforms.functional import to_tensor, resize

from kornia.metrics import ssim as ssim_metric
from kornia.metrics import psnr as psnr_metric
import time
import numpy as np

class StructuralSimilarityIndex:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_metric(img1, img2, window_size=self.window_size).mean()


class PeakSignalNoiseRatio:
    def __init__(self, max_value: float = 1):
        self.max_value = max_value

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return psnr_metric(img1,img2, max_val=self.max_value)





def paths_from_experiment(args):
    experiment_name = args.experiment
    resolution = args.resolution
    tile = args.tileSize
    step = args.step
    if resolution != "full":
        resolution = "crop-" + resolution

    base = f"results/{experiment_name}/test_tile-{tile}_{resolution}_{step}/images"
    target = base + "/gt"
    candidate = base +"/synthesized_image"
    return candidate, target

def store_metrics(args, filenames, results):
    experiment_name = args.experiment
    resolution = args.resolution
    tile = args.tileSize
    step = args.step
    psnrs, ssims = results
    if resolution != "full":
        resolution = "crop-" + resolution
    result = f"""PSNR: mean: {psnrs.mean():.3f} | std: {psnrs.std():.3f}
SSIM: mean: {ssims.mean():.3f} | std: {ssims.std():.3f}"""
    base = f"results/{experiment_name}/test_tile-{tile}_{resolution}_{step}/{len(psnrs)}-metrics.txt"
    with open(base, "a") as f:
        now = time.strftime("%c")
        f.write(f'\n================ {now} ================\n' )
        f.write("file, psnr, ssim")
        for i, filename in enumerate(filenames):
            name = filename.split("/")[-1]
            psnr = psnrs[i].item()
            ssim = ssims[i].item()
            f.write(f"{name}, {psnr}, {ssim}\n")
        f.write(f'\n================  ================\n' )
        f.write(result)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test with damfs data')
    parser.add_argument("--files", "-f", type=str, default=None)
    parser.add_argument("--ground_truth", "-gt", type=str, default=None)
    parser.add_argument("--experiment", "-e", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resolution", type=str, default="full")
    parser.add_argument("--step", type=str, default="latest")
    parser.add_argument("--tileSize", type=str, default="1024")
    parser.add_argument("--half", action='store_true', default=False)
    parser.add_argument("--no_strict", action='store_true', default=False)
    parser.add_argument("--how_many", type=float, default=float("inf"))
    parser.add_argument("--store", action='store_true', default=False)
    parser.add_argument("--deeplpf_match", action='store_true', default=False)
    
    args = parser.parse_args()


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device(torch.cuda.current_device())
        print("Using cuda device:", device)
    else:
        device = torch.device("cpu")

    psnrs = list()
    ssims = list()


    compute_ssim = StructuralSimilarityIndex()
    compute_psnr = PeakSignalNoiseRatio()
    

    if args.experiment is None:
        path_a, path_b = args.files, args.ground_truth
    else:
        path_a, path_b = paths_from_experiment(args)

    print(f"Reading files from {path_a}")
    paths = list(Path(path_a).glob("**/*.jpg"))
    print(f"Matching with {path_b}")
    result = []
    for file_path in paths:
        file_a = file_path.as_posix()
        if args.deeplpf_match:
           name = file_a.split("/")[-1]
           id = name.split("_")[0]
           file_b = os.path.join(path_b, f"{id}.jpg")
        else:
            file_b = file_a.replace(path_a, path_b)

        if os.path.exists(file_b):
            result.append((file_a, file_b))
    
    print(f"{len(result)} files to be compared with {args.tileSize} at {args.resolution} res from {args.step} step.")
    
    tqbar = tqdm(result, ncols=110)

    processed = 0
    names = []
    for file_a, file_b in tqbar:
        names.append(file_a)
        image_a, image_b = Image.open(file_a), Image.open(file_b)
        tensor_a, tensor_b = to_tensor(image_a).to(device).unsqueeze(0), to_tensor(image_b).to(device).unsqueeze(0)
        if tensor_a.shape != tensor_b.shape and args.no_strict:
            tensor_b = resize(tensor_b, tensor_a.shape[-2:])
        

        
        

        psnr = compute_psnr(tensor_a.clone(), tensor_b.clone()).cpu()
        ssim = compute_ssim(tensor_a.clone(), tensor_b.clone()).cpu()

        psnrs.append(psnr)
        ssims.append(ssim)

        tqbar.set_description(f"PSNR: {torch.tensor(psnrs).float().mean():.3f}| SSIM: {torch.tensor(ssims).float().mean():.3f}")
        torch.cuda.empty_cache()
        processed += 1
        if processed >= args.how_many:
            break

    # Summary
    psnrs = torch.tensor(psnrs).float()
    ssims = torch.tensor(ssims).float()
    result = f"""PSNR: mean: {psnrs.mean():.3f} | std: {psnrs.std():.3f}
SSIM: mean: {ssims.mean():.3f} | std: {ssims.std():.3f}"""
    if args.store:
        store_metrics(args, names, (psnrs, ssims))
    print(result)
    
