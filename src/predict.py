from ultralytics import YOLO
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained YOLO model.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, folder, video, or webcam index"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="assets",
        help="Output base folder"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predictions",
        help="Output folder name inside project"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = args.source

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=source_path,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    print(f"Inference completed. Results saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()