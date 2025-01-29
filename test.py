import os
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###########################
    # Load Model
    ###########################
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
    model.linear_head = nn.Sequential(
        nn.Linear(5120, 1024),  
        nn.ELU(),             
        nn.Dropout(p=0.3),     
        nn.Linear(1024, 512),
        nn.ELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 20)
    )
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint_dir + "/best.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Using model at epoch", checkpoint["epoch"])

    ###########################
    # Random TTA Transform
    ###########################
    # We'll apply this transform multiple times per image for TTA.
    tta_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (392, 392), scale=(0.8, 1.0), interpolation=F.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    ###########################
    # Center-Crop Transform
    ###########################
    # Resize to 448×448, then center-crop to 392×392.
    center_transform = transforms.Compose(
        [
            transforms.Resize((448, 448), interpolation=F.InterpolationMode.BILINEAR),
            transforms.CenterCrop(392),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    ###########################
    # TTA Inference Function
    ###########################
    def tta_inference(
        image: Image.Image, model: nn.Module, device: torch.device, num_augs: int = 5
    ) -> torch.Tensor:
        """
        - Applies 'num_augs' random transformations.
        - Applies one center-crop transform (zoom-in style).
        - Averages the outputs from all transforms and returns the averaged logits.
        """
        outputs = []

        with torch.no_grad():
            # (1) Random transforms
            for _ in range(num_augs):
                aug_img = tta_transform(image)
                aug_img = aug_img.unsqueeze(0).to(device)  # [1, C, H, W]
                out = model(aug_img)  # [1, num_classes]
                outputs.append(out)

            # (2) Center-crop transform
            cc_img = center_transform(image)
            cc_img = cc_img.unsqueeze(0).to(device)  # [1, C, H, W]
            out_cc = model(cc_img)  # [1, num_classes]
            outputs.append(out_cc)

        # Stack & average all outputs: shape => [num_augs + 1, num_classes]
        outputs = torch.cat(outputs, dim=0)
        avg_output = outputs.mean(dim=0, keepdim=True)  # [1, num_classes]
        return avg_output

    ###########################
    # Inference Loop
    ###########################
    TEST_DIR = "/gpfs/workdir/yutaoc/bird/bird_dataset/test_images/mistery_cat"
    predictions = []

    for file_name in os.listdir(TEST_DIR):
        file_path = os.path.join(TEST_DIR, file_name)
        if not os.path.isfile(file_path):
            continue  # skip directories, hidden files, etc.

        image = Image.open(file_path).convert("RGB")

        # Perform TTA inference
        output = tta_inference(image, model, device, num_augs=5)
        cls = torch.argmax(output, dim=1).item()
        predictions.append([file_name, cls])

    df = pd.DataFrame(predictions, columns=["path", "class_idx"])
    df.to_csv(f"{args.checkpoint_dir}/submission.csv", index=False)
    print(f"Predictions saved to {args.checkpoint_dir}/submission.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Inference")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/dino_combined",
        type=str,
        help="checkpoint path",
    )
    args = parser.parse_args()
    main(args)
