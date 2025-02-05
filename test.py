import os
from PIL import Image
import pandas as pd

import timm
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###########################
    # Load Model
    ###########################
    model = timm.create_model(args.model, pretrained=True)
    # Example head definition
    model.head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 20)
    )
    model = model.to(device)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best.pth"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Using model at epoch", checkpoint["epoch"])

    ###############################################################################
    # All transforms produce a final size of 448×448 to match the model’s input size
    ###############################################################################

    ##############
    # Center-Crop (Normal)
    ##############
    # Resize so the *shorter side* is 448, then center-crop 448×448
    center_transform = transforms.Compose([
        transforms.Resize(448, interpolation=F.InterpolationMode.BILINEAR),  # shorter side = 448
        transforms.CenterCrop(448),  # final 448×448
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758]
        ),
    ])

    ##############
    # Center-Crop (Flipped)
    ##############
    # Same as above but with forced horizontal flip
    center_transform_flip = transforms.Compose([
        transforms.Resize(448, interpolation=F.InterpolationMode.BILINEAR),
        transforms.CenterCrop(448),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758]
        ),
    ])

    ##############
    # Multi-Scale (Normal) Helper
    ##############
    # For each scale s ≥ 448: 
    #   1) Resize(shorter_side=s)
    #   2) CenterCrop(448) 
    # => final 448×448
    def scale_center_transform(image: Image.Image, s: int) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(s, interpolation=F.InterpolationMode.BILINEAR),
            transforms.CenterCrop(448),  # 448×448 final
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4815, 0.4578, 0.4082],
                std=[0.2686, 0.2613, 0.2758]
            ),
        ])
        return transform(image)

    ##############
    # Multi-Scale (Flipped) Helper
    ##############
    def scale_center_transform_flip(image: Image.Image, s: int) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(s, interpolation=F.InterpolationMode.BILINEAR),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4815, 0.4578, 0.4082],
                std=[0.2686, 0.2613, 0.2758]
            ),
        ])
        return transform(image)

    ###########################
    # TTA Inference Function
    ###########################
    def tta_inference(
        image: Image.Image,
        model: nn.Module,
        device: torch.device,
        scales: list = [448, 512, 576, 640],
    ) -> torch.Tensor:
        """
        Applies:
          1) Center-crop (normal + flip)
          2) Multi-scale (normal + flip), each producing a 448×448 final image.

        Averages all logits for the final output.
        """
        outputs = []

        with torch.no_grad():
            # (1) Center-Crop Normal
            cc_img = center_transform(image).unsqueeze(0).to(device)
            out_cc = model(cc_img)
            outputs.append(out_cc)

            # (2) Center-Crop Flipped
            cc_img_flip = center_transform_flip(image).unsqueeze(0).to(device)
            out_cc_flip = model(cc_img_flip)
            outputs.append(out_cc_flip)

            # (3) Multi-Scale (Normal + Flipped)
            for s in scales:
                # Normal
                sc_img = scale_center_transform(image, s).unsqueeze(0).to(device)
                out_sc = model(sc_img)
                outputs.append(out_sc)

                # Flipped
                sc_img_flip = scale_center_transform_flip(image, s).unsqueeze(0).to(device)
                out_sc_flip = model(sc_img_flip)
                outputs.append(out_sc_flip)

        # Combine all results and average
        # total passes: 2 (center-crop) + 2×len(scales) (multi-scale) = 2 + 2N
        outputs = torch.cat(outputs, dim=0)  # shape [N, num_classes]
        avg_output = outputs.mean(dim=0, keepdim=True)  # shape [1, num_classes]
        return avg_output

    ###########################
    # Inference Loop
    ###########################
    TEST_DIR = "/gpfs/workdir/yutaoc/bird/bird_dataset/test_images/mistery_cat"
    predictions = []

    for file_name in tqdm(os.listdir(TEST_DIR), desc="Inference"):
        file_path = os.path.join(TEST_DIR, file_name)
        if not os.path.isfile(file_path):
            continue

        image = Image.open(file_path).convert("RGB")

        # TTA Inference
        output = tta_inference(
            image, 
            model, 
            device, 
            scales=[448, 512, 576, 640]
        )
        cls = torch.argmax(output, dim=1).item()
        predictions.append([file_name, cls])

    # Save predictions
    df = pd.DataFrame(predictions, columns=["path", "class_idx"])
    df.to_csv(os.path.join(args.checkpoint_dir, "submission.csv"), index=False)
    print(f"Predictions saved to {os.path.join(args.checkpoint_dir, 'submission.csv')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Inference")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/dino_combined",
        type=str,
        help="Checkpoint directory with best.pth",
    )
    parser.add_argument(
        "--model",
        default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        type=str,
    )
    args = parser.parse_args()
    main(args)