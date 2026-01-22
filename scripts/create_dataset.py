#!/usr/bin/env python3
"""Script to create toy datasets for testing."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import create_toy_dataset


def main():
    """Main function to create toy dataset."""
    parser = argparse.ArgumentParser(description="Create toy dataset for few-shot object detection")
    parser.add_argument("--output_dir", type=str, default="data/toy_dataset", 
                       help="Output directory for dataset")
    parser.add_argument("--num_classes", type=int, default=5, 
                       help="Number of classes")
    parser.add_argument("--samples_per_class", type=int, default=100, 
                       help="Number of samples per class")
    
    args = parser.parse_args()
    
    print(f"Creating toy dataset with {args.num_classes} classes and "
          f"{args.samples_per_class} samples per class...")
    
    create_toy_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class
    )
    
    print(f"Dataset created successfully in {args.output_dir}")


if __name__ == "__main__":
    main()
