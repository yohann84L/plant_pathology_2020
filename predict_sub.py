import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import models.tta as tta
from datasets import PlantPathologyDataset, DatasetTransformsAutoAug
from utils.utils import str2bool, get_device, load_for_inference
from torch2trt import torch2trt

def parse_args():
    """
    Parse input arguments
    Returns:
        args
    """
    parser = argparse.ArgumentParser(description="Train Faster R-CNN model on TrayVisor dataset")
    parser.add_argument("--checkpoint", dest="checkpoint",
                        help="checkpoint for inference", type=str)
    parser.add_argument("--img_root", dest="img_root",
                        help="training dataset", type=str)
    parser.add_argument("--annot_test", dest="annot_test",
                        help="annotation test df", type=str)
    parser.add_argument("--sample_sub", dest="sample_sub",
                        help="sample submission", type=str)
    parser.add_argument("--use_cuda", dest="use_cuda",
                        help="use cuda or not", type=str2bool, nargs="?",
                        default=True)
    parser.add_argument("--tta", dest="tta",
                        help="use tta or not", type=str,
                        default="d4_image2label")
    parser.add_argument("--img_size", dest="img_size",
                        help="resize img to the given int", type=int,
                        nargs=2, default=-1)
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        default=4, help="batch size inference")
    parser.add_argument("--user_ttr", dest="use_ttr", type=str2bool,
                        nargs="?", default=False)
    args = parser.parse_args()
    return args


@torch.no_grad()
def predict_submission(model, annot_test_fp: str, sample_sub_fp: str, img_root: str, use_tta: str,
                       img_size: iter = None, batch_size: int = 4, submission_name: str = None):
    if submission_name is None:
        submission_name = "sub.csv"

    test_transforms = DatasetTransformsAutoAug(train=False, img_size=img_size)

    submission_df = pd.read_csv(sample_sub_fp)
    dataset_test = PlantPathologyDataset(annot_fp=annot_test_fp, img_root=img_root,
                                         transforms=test_transforms)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    device = get_device(use_cuda=True)
    preds = None

    bar = tqdm(total=len(data_loader_test.dataset))
    for imgs, _ in data_loader_test:
        imgs = imgs.to(device, dtype=torch.float)

        if use_tta:
            outputs = tta.d4_image2label(model, imgs)
        else:
            outputs = model(imgs)

        if preds is None:
            preds = outputs.data.cpu()
        else:
            preds = torch.cat((preds, outputs.data.cpu()), dim=0)

        bar.update(batch_size)

    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(preds, dim=1)
    submission_df.to_csv(submission_name, index=False)


def tta_available():
    tta_avail = [
        "d4_image2label",
        "fivecrop_image2label"
    ]
    print("TTA arg should be one of the following:")
    for tta in tta_avail:
        print(f"     - {tta}")


if __name__ == '__main__':
    args = parse_args()
    model = load_for_inference(args.checkpoint)

    if args.use_ttr:
        x = torch.ones((1, 3, args.img_size[0], args.img_size[1])).to(get_device(args.use_cuda))
        model = torch2trt(model, x)

    predict_submission(
        model=model,
        annot_test_fp=args.annot_test,
        sample_sub_fp=args.sample_sub,
        img_root=args.img_root,
        use_tta=args.tta,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
