import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

# Add src to path if package not installed in editable mode, though uv/standard install should handle this.
# For script execution without install:
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import from pangu package
# Note: These components must exist in src/pangu/models/ and src/pangu/training/
# If they don't exist yet, we define placeholders here for the script to be valid syntax
try:
    from pangu.models import PanguModel
    from pangu.training import TrainingConfig  # Hypothetical config class
except ImportError:
    print(
        "Warning: pangu package imports failed. Using placeholders for demonstration."
    )

    class PanguModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)  # Dummy

        def forward(self, x, x_surface):
            return x, x_surface


class WeatherDataset(Dataset):
    def __init__(self, data_dir, start_year, end_year):
        self.data_dir = data_dir
        # TODO: Implement actual data loading logic from .npy or .grib files
        # mimicking LoadData(step) from pseudocode
        self.length = 100  # Dummy length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Placeholder data generation
        # input: (5, 13, 721, 1440), input_surface: (4, 721, 1440)
        # return input, input_surface, target, target_surface
        input_tensor = torch.randn(5, 13, 10, 20)  # Reduced size for dummy
        input_surface = torch.randn(4, 10, 20)
        target = torch.randn(5, 13, 10, 20)
        target_surface = torch.randn(4, 10, 20)
        return input_tensor, input_surface, target, target_surface


def train(args):
    print(f"Starting training with config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Config (if exists)
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Initialize Model
    model = PanguModel().to(device)

    # Optimizer
    # Pseudocode: UpdateModelParametersWithAdam(lr=5e-4, weight_decay=3e-6)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)

    # Loss Function
    # Pseudocode: TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25
    criterion = nn.L1Loss()

    # Data Loader
    train_dataset = WeatherDataset(args.data_dir, 1979, 2017)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (input, input_surf, target, target_surf) in enumerate(train_loader):
            input = input.to(device)
            input_surf = input_surf.to(device)
            target = target.to(device)
            target_surf = target_surf.to(device)

            optimizer.zero_grad()

            output, output_surf = model(input, input_surf)

            # Weighted Loss
            loss_main = criterion(output, target)
            loss_surf = criterion(output_surf, target_surf)
            loss = loss_main + loss_surf * 0.25

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(train_loader)}")

        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"pangu_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Pangu-Weather Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/input_data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
