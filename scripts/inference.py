#!/usr/bin/env python
"""
Inference script for Pangu-Weather ONNX models.

Usage:
    python scripts/inference.py --config configs/inference.yaml --lead-time 24h
    python scripts/inference.py --iterative --hours 168
"""

import argparse
import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pangu.inference.onnx_engine import PanguInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run Pangu-Weather inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--lead-time",
        type=str,
        default="24h",
        choices=["1h", "3h", "6h", "24h"],
        help="Lead time for single forecast",
    )
    parser.add_argument(
        "--iterative", action="store_true", help="Run iterative forecast"
    )
    parser.add_argument(
        "--hours", type=int, default=168, help="Total forecast hours for iterative mode"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save intermediate results every N hours",
    )

    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
        inference_config = full_config.get("inference", {})

    # Initialize engine
    print(f"Initializing Pangu Inference Engine...")
    engine = PanguInferenceEngine(config=inference_config)

    # Load input
    input_dir = inference_config.get("input_dir", "data/input_data")
    output_dir = inference_config.get("output_dir", "data/output_data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading input data from {input_dir}...")
    input_upper = np.load(os.path.join(input_dir, "input_upper.npy")).astype(np.float32)
    input_surface = np.load(os.path.join(input_dir, "input_surface.npy")).astype(
        np.float32
    )

    print(f"Input shapes: upper={input_upper.shape}, surface={input_surface.shape}")

    if args.iterative:
        print(f"Running iterative forecast for {args.hours} hours...")

        results_generator = engine.run_sequence(
            input_upper, input_surface, total_hours=args.hours
        )

        count = 0
        for hour, out_upper, out_surface in results_generator:
            count += 1
            # Save if needed
            if args.save_every and hour % args.save_every == 0:
                save_results(output_dir, out_upper, out_surface, suffix=f"_{hour}h")
                print(f"Step {count}: Saved {hour}h forecast")

        # Save final
        save_results(output_dir, out_upper, out_surface, suffix=f"_final_{hour}h")
        print(f"Iterative forecast complete. Total steps: {count}")

    else:
        # Single forecast
        print(f"Running single {args.lead_time} forecast...")
        output_upper, output_surface = engine.predict(
            input_upper, input_surface, args.lead_time
        )
        save_results(output_dir, output_upper, output_surface)
        print("Forecast complete!")


def save_results(out_dir, upper, surface, suffix=""):
    np.save(os.path.join(out_dir, f"output_upper{suffix}.npy"), upper)
    np.save(os.path.join(out_dir, f"output_surface{suffix}.npy"), surface)


if __name__ == "__main__":
    main()
