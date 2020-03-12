import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PlantPathologyDataset, DatasetTransforms
from utils.utils import str2bool, get_device, load_for_inference
import models.tta as tta

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
    args = parser.parse_args()
    return args


@torch.no_grad()
def predict_submission(model, annot_test_fp: str, sample_sub_fp: str, img_root: str, submission_name: str = None,
                       use_tta: bool = True):
    if submission_name is None:
        submission_name = "sub.csv"

    submission_df = pd.read_csv(sample_sub_fp)
    dataset_test = PlantPathologyDataset(annot_fp=annot_test_fp, img_root=img_root,
                                         transforms=DatasetTransforms(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    device = get_device(use_cuda=True)

    for i, batch in enumerate(tqdm(data_loader_test)):
        images = batch[0]
        images = images.to(device, dtype=torch.float)

        if use_tta:
            outputs = tta.d4_image2label(model, images)
        else:
            outputs = model(images)

        preds = torch.nn.functional.softmax(outputs)
        preds = preds.cpu().detach().numpy()

        submission_df.iloc[i, 1:] = preds[0]
    submission_df.to_csv(submission_name, index=False)


if __name__ == '__main__':
    args = parse_args()
    model = load_for_inference(args.checkpoint)
    predict_submission(model, args.annot_test, args.sample_sub, args.img_root, use_tta=True)
