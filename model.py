import math
from typing import Tuple, Optional, Callable, Union
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# Set random seeds for reproducibility


class ElectronOpticsDataset(Dataset):
    """Dataset class for electron optics data"""

    def __init__(self, voltages: np.ndarray, values: np.ndarray):
        self.voltages = torch.FloatTensor(voltages)
        self.values = torch.FloatTensor(values)

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        return self.voltages[idx], self.values[idx]


class ElectronOpticsModel(nn.Module):
    """Neural network model for predicting electron optics values from voltages"""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dims: list = [32, 64, 128, 256, 128, 64, 32],
        leak: float = 0.0,
    ):
        super(ElectronOpticsModel, self).__init__()
        self.hidden_dims = hidden_dims
        # Build the network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(leak),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (vector output)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.leak = leak
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ElectronOpticsPredictor:
    """Main class for training and using the electron optics prediction model"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        device: Optional[str] = None,
        leak: float = 0.0,
    ):
        device = self.get_device() if device is None else device
        self.device = torch.device(device)

        self.model = ElectronOpticsModel(input_dim, output_dim, leak=leak).to(
            self.device
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler_voltages = None
        self.scaler_values = None
        self.train_ds = None
        self.validation_ds = None

        print(f"Using device: {self.device}")

    def _normalize_data(
        self, data: np.ndarray, scaler: Optional[dict] = None, fit: bool = False
    ):
        """Normalize data using min-max scaling"""
        if fit or (scaler is None):
            scaler = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}

        # Avoid division by zero
        range_vals = scaler["max"] - scaler["min"]
        range_vals[range_vals == 0] = 1
        if isinstance(data, torch.Tensor):
            normalized = (
                data
                - torch.tensor(scaler["min"], device=self.device, dtype=torch.float32)
            ) / torch.tensor(range_vals, device=self.device, dtype=torch.float32)
        else:
            normalized = (data - scaler["min"]) / range_vals
        return normalized, scaler

    def _denormalize_values(self, normalized_values: np.ndarray):
        """Denormalize values back to original scale"""
        if self.scaler_values is None:
            return normalized_values
        if isinstance(normalized_values, torch.Tensor):
            denormalized = normalized_values * (
                torch.tensor(
                    (self.scaler_values["max"] - self.scaler_values["min"]),
                    device=self.device,
                    dtype=torch.float32,
                )
            ) + torch.tensor(
                self.scaler_values["min"], device=self.device, dtype=torch.float32
            )
        else:
            denormalized = (
                normalized_values
                * (self.scaler_values["max"] - self.scaler_values["min"])
                + self.scaler_values["min"]
            )

        return denormalized

    def _denormalize_voltages(self, normalized_voltages: np.ndarray):
        """Denormalize voltages back to original scale"""
        if self.scaler_voltages is None:
            return normalized_voltages

        range_vals = self.scaler_voltages["max"] - self.scaler_voltages["min"]
        return normalized_voltages * range_vals + self.scaler_voltages["min"]

    def train(
        self,
        voltages: np.ndarray,
        values: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True,
        weight_name: str = "best_model.pth",
    ):
        """Train the model on voltage-value pairs"""

        # Ensure values has shape (n_samples, output_dim)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # Normalize the data
        voltages_norm, self.scaler_voltages = self._normalize_data(voltages, fit=True)
        values_norm, self.scaler_values = self._normalize_data(values, fit=True)

        # Split data
        n_samples = len(voltages_norm)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_dataset = ElectronOpticsDataset(
            voltages_norm[train_idx], values_norm[train_idx]
        )
        val_dataset = ElectronOpticsDataset(
            voltages_norm[val_idx], values_norm[val_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.train_ds = train_dataset
        self.val_ds = val_dataset

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=1e-4)

        # Try OneCycleLR instead of ReduceLROnPlateau
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        self.scheduler = scheduler.__class__.__name__
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        batch_voltages: torch.Tensor
        batch_values: torch.Tensor
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_voltages, batch_values in train_loader:
                batch_voltages = batch_voltages.to(self.device)
                batch_values = batch_values.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_voltages)
                loss: torch.Tensor = criterion(outputs, batch_values)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_voltages, batch_values in val_loader:
                    batch_voltages = batch_voltages.to(self.device)
                    batch_values = batch_values.to(self.device)

                    outputs = self.model(batch_voltages)
                    loss = criterion(outputs, batch_values)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                # J: Why save here without self.save?
                torch.save(self.model.state_dict(), weight_name)
            else:
                patience_counter += 1
                if patience_counter > 1000:  # Early stopping patience
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )

        # Load best model
        self.model.load_state_dict(torch.load(weight_name))

        if verbose:
            print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.yscale("log")
            plt.title("Training Progress")
            plt.annotate(
                f"hidden_dims={self.model.hidden_dims}\nscheduler={self.scheduler}\nN={len(train_dataset)}\nleak={self.model.leak}",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"
                ),
            )
            plt.show()

    @staticmethod
    def get_device() -> str:

        # Check for available devices in order of preference: MPS, CUDA, CPU
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def predict(
        self,
        voltages: Union[np.ndarray, torch.Tensor],
        require_grad: bool = False,
        as_numpy: bool = True,  # ← return NumPy by default for inference
    ):
        """
        If `require_grad=True`, returns a **torch.Tensor** on the same device
        so that gradients can flow.  Otherwise returns NumPy (unless you set
        `as_numpy=False`).
        """
        self.model.eval()

        # -------- 1. Convert input to a tensor on the right device ----------
        if isinstance(voltages, np.ndarray):
            voltages_tensor = torch.as_tensor(
                voltages, dtype=torch.float32, device=self.device
            )
        else:
            voltages_tensor = voltages.to(self.device, dtype=torch.float32)

        voltages_tensor = voltages_tensor.flatten()

        # -------- 2. Normalise (pure-torch math so graph is intact) ---------
        vmin = torch.as_tensor(
            self.scaler_voltages["min"], dtype=torch.float32, device=self.device
        )
        vmax = torch.as_tensor(
            self.scaler_voltages["max"], dtype=torch.float32, device=self.device
        )
        voltages_norm = (voltages_tensor - vmin) / (vmax - vmin)

        voltages_norm.requires_grad_(require_grad)

        # -------- 3. Forward pass ------------------------------------------
        with torch.set_grad_enabled(require_grad):
            preds_norm: torch.Tensor = self.model(
                voltages_norm.unsqueeze(0)
            )  # (1, n_out)

            # denormalise **in torch**
            if self.scaler_values is not None:
                smin = torch.as_tensor(
                    self.scaler_values["min"], dtype=torch.float32, device=self.device
                )
                smax = torch.as_tensor(
                    self.scaler_values["max"], dtype=torch.float32, device=self.device
                )
                preds = preds_norm * (smax - smin) + smin
            else:
                preds = preds_norm

        preds = preds.squeeze(0)  # shape (n_out,)

        # -------- 4. Decide return type ------------------------------------
        if require_grad or not as_numpy:
            return preds  # torch.Tensor with grad_fn (if any)
        else:
            return preds.detach().cpu().numpy()

    def optimize_voltages(
        self,
        objective_function: Callable = None,
        value_index: int = 0,
        weights: list = None,
        n_iterations: int = 1000,
        learning_rate: float = 0.1,
        random_restarts: int = 5,
        voltage_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        constrain_to_training_range: bool = False,
    ):
        """Find voltages that optimize the predicted values according to a custom objective

        Args:
            objective_function: Custom function that takes the model prediction tensor and returns a scalar to maximize
                                Example: lambda pred: pred[:, 0] - 0.5 * pred[:, 1]
            value_index: Index of the value to optimize if optimizing a single output
            weights: Weights for each output value for weighted optimization;
                     positive for maximization, negative for minimization
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            random_restarts: Number of random starting points to try
            voltage_bounds: Tuple of (min_voltages, max_voltages) to constrain optimization
            constrain_to_training_range: If True, constrains voltages to the range seen during training.
                                         If False, allows exploration beyond training data range.

        Returns:
            best_voltages: Optimal voltage settings
            best_values: Predicted values at the optimal voltage settings
            best_objective: Value of the objective function at the optimal point
        """
        return optimize_voltages(
            [self],
            objective_function,
            value_index,
            weights,
            n_iterations,
            learning_rate,
            random_restarts,
            voltage_bounds,
            constrain_to_training_range,
        )

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler_voltages": self.scaler_voltages,
                "scaler_values": self.scaler_values,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "leak": self.model.leak,
                "train_ds": self.train_ds,
                "validation_ds": self.validation_ds,
            },
            filepath,
        )

    @classmethod
    def load_model(cls, filepath: str, device: Optional[str] = None):
        """Load a trained model"""
        if device is None:
            device = cls.get_device()
        checkpoint: dict = torch.load(
            filepath, map_location=torch.device(device), weights_only=False
        )

        # Recreate model with proper dimensions
        predictor = cls(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint.get("output_dim", 1),
            device=device,
            leak=checkpoint.get("leak", 0.0),
        )
        # Load state and scalers
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.scaler_voltages = checkpoint["scaler_voltages"]
        predictor.scaler_values = checkpoint["scaler_values"]

        return predictor


def optimize_voltages(
    predictors: list[ElectronOpticsPredictor],
    objective_function: Optional[Callable] = None,
    value_index: int = 0,
    weights: list = None,
    n_iterations: int = 1000,
    learning_rate: float = 0.1,
    random_restarts: int = 5,
    voltage_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    constrain_to_training_range: bool = False,
):
    """Find voltages that optimize the predicted values according to a custom objective

    Args:
        objective_function: Custom function that takes the model prediction tensor and returns a scalar to maximize
                            Example: lambda pred: pred[:, 0] - 0.5 * pred[:, 1]
        value_index: Index of the value to optimize if optimizing a single output
        weights: Weights for each output value for weighted optimization;
                    positive for maximization, negative for minimization
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        random_restarts: Number of random starting points to try
        voltage_bounds: Tuple of (min_voltages, max_voltages) to constrain optimization
        constrain_to_training_range: If True, constrains voltages to the range seen during training.
                                        If False, allows exploration beyond training data range.


    Returns:
        best_voltages: Optimal voltage settings
        best_values: Predicted values at the optimal voltage settings
        best_objective: Value of the objective function at the optimal point
    """
    best_voltages = None
    best_values = None
    best_objective = float("inf")
    device = predictors[0].device

    # Define the objective function
    if objective_function is None:
        if weights is not None:
            # Weighted sum of values
            def objective_func(pred):
                return torch.sum(
                    pred * torch.tensor(weights, device=device, dtype=torch.float32)
                )

        else:
            # Maximize single value
            def objective_func(pred):
                return pred[:, value_index]

    else:
        # Use the provided custom objective function
        objective_func = objective_function
    plt.figure(figsize=(10, 5))
    for restart in range(random_restarts):
        losses = []
        # Initialize voltages
        if voltage_bounds is not None:
            voltages = np.random.uniform(voltage_bounds[0], voltage_bounds[1])
        else:
            voltages = np.random.uniform(
                predictors[0].scaler_voltages["min"],
                predictors[0].scaler_voltages["max"],
                size=predictors[0].input_dim,
            )

        # Normalize initial voltages

        voltages = voltages.flatten()

        # Convert to tensor and require gradients
        voltages_tensor = torch.FloatTensor(voltages).to(device)
        voltages_tensor.requires_grad_(True)

        optimizer = optim.Adam([voltages_tensor], lr=learning_rate)

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Predict values
            predictions = torch.empty(0, dtype=torch.float32, device=device)

            for predictor in predictors:
                prediction = predictor.predict(
                    voltages_tensor.unsqueeze(0)[:, : predictor.input_dim],
                    require_grad=True,
                )
                prediction = torch.Tensor(prediction).squeeze(0).to(device)
                predictions = torch.cat((predictions, prediction), dim=0)

            predictions = predictions.to(device)

            # Compute objective (negative for maximization)
            loss = objective_func(predictions)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Optionally constrain to training range
            if constrain_to_training_range:
                with torch.no_grad():
                    voltages_tensor.clamp_(0, 1)

        plt.plot(losses, label=f"Train Loss (Restart {restart+1})")

        # Get final result
        with torch.no_grad():
            final_predictions = torch.empty(0, dtype=torch.float32, device=device)
            for predictor in predictors:
                final_prediction = predictor.predict(
                    voltages_tensor.unsqueeze(0), require_grad=True
                )
                final_predictions = torch.cat(
                    (final_predictions, final_prediction), dim=0
                )
            final_predictions = final_predictions.to(device)

            final_objective = objective_func(final_predictions).item()

            if final_objective < best_objective:
                best_objective = final_objective
                best_voltages = voltages_tensor.cpu().numpy()
                best_values = final_predictions
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Voltage Optimization Progress")
    plt.show()
    return (
        best_voltages,
        best_values,
        best_objective,
    )  # best_values is best predicted output_values and best_objective is best metric value.


def custom_objective(pred):
    """
    Custom objective function to optimize.
    Takes predicted values and returns a scalar to maximize.

    You define your own objective function based on your electron optics requirements.
    """
    # This is just a placeholder - replace with your actual objective
    if isinstance(pred, torch.Tensor):
        return (
            pred[0] ** 2
            + (torch.abs(pred[1]) - 0.1) ** 2
            + pred[2] ** 2
            + pred[3] ** 2
            + pred[4] ** 2
            + pred[5] ** 2
            + (torch.abs(pred[6]) - 500) ** 2
            + pred[7] ** 2
            + pred[8] ** 2
            + pred[9] ** 2
            + pred[10] ** 2
            + pred[11] ** 2
        )
    else:
        return (
            pred[0] ** 2
            + (np.abs(pred[1]) - 0.1) ** 2
            + pred[2] ** 2
            + pred[3] ** 2
            + pred[4] ** 2
            + pred[5] ** 2
            + (np.abs(pred[6]) - 500) ** 2
            + pred[7] ** 2
            + pred[8] ** 2
            + pred[9] ** 2
            + pred[10] ** 2
            + pred[11] ** 2
        )


device = ElectronOpticsPredictor.get_device()  # 'mps' or 'cpu'
dtype = torch.float32

solid_angle_scaling = torch.tensor(
    math.sin(math.radians(5 / 2)) / math.sin(math.radians(60 / 2)),
    device=device,
    dtype=dtype,
)
APER_0_D = torch.tensor(0.5, device=device, dtype=dtype)
DET_D = torch.tensor(25.0, device=device, dtype=dtype)


# ---------- all-torch versions ----------
def angle_resolved_aper0(aper0_map: torch.Tensor) -> torch.Tensor:
    """aper0_map shape: (6,) tensor on *any* device"""
    aper0_map = aper0_map.to(device)  # move to same GPU/CPU
    aberr = (
        aper0_map[0] ** 2
        + aper0_map[2] ** 2
        + (aper0_map[3] * solid_angle_scaling) ** 2
        + (aper0_map[4] * solid_angle_scaling**2) ** 2
        + (aper0_map[5] * solid_angle_scaling**3) ** 2
    )

    return aberr + (torch.abs(aper0_map[1] * solid_angle_scaling) - APER_0_D / 2) ** 2


def spatial_resolved_detector(det_map: torch.Tensor) -> torch.Tensor:
    """det_map shape: (≥6,) tensor on *any* device"""
    det_map = det_map.to(device)
    aberr = (
        (det_map[1] * solid_angle_scaling) ** 2
        + det_map[2] ** 2
        + (det_map[3] * solid_angle_scaling) ** 2
        + (det_map[4] * solid_angle_scaling**2) ** 2
        + (det_map[5] * solid_angle_scaling**3) ** 2
    )

    return aberr + (torch.abs(det_map[0]) - DET_D / 2) ** 2


def metric(output: torch.Tensor) -> torch.Tensor:
    """`output` is the concatenated predictor output tensor."""
    return angle_resolved_aper0(output[:6]) * 1_000 + spatial_resolved_detector(
        output[6:]
    )


"""
n_voltages = 14     # Number of voltage parameters
n_output_values = 2 # Number of output values (e.g., magnification, aberration, etc.)




# Example optimization using a custom objective function
print("\nOptimizing voltages...")
voltage_bounds = (np.full(n_voltages, -10), np.full(n_voltages, 10))

# Define a custom objective function
# Replace this with your own function that defines what "good" means for your system

# Run optimization allowing exploration beyond training range
optimal_voltages, optimal_values, obj_value = predictor.optimize_voltages(
    objective_function=custom_objective,
    n_iterations=1000,
    learning_rate=0.01,
    random_restarts=5,
    voltage_bounds=voltage_bounds,
    constrain_to_training_range=False  # Allow exploration beyond training data range
)

print(f"\nOptimal voltages found (unconstrained):")
print(f"Voltages: {optimal_voltages}")
print(f"Predicted values: {optimal_values}")
print(f"Objective value: {obj_value:.4f}")

# Also try optimization with constraining to training range for comparison
optimal_voltages_constrained, optimal_values_constrained, obj_value_constrained = predictor.optimize_voltages(
    objective_function=custom_objective,
    n_iterations=1000,
    learning_rate=0.01,
    random_restarts=5,
    voltage_bounds=voltage_bounds,
    constrain_to_training_range=True  # Constrain to training data range
)

print(f"\nOptimal voltages found (constrained to training range):")
print(f"Voltages: {optimal_voltages_constrained}")
print(f"Predicted values: {optimal_values_constrained}")
print(f"Objective value: {obj_value_constrained:.4f}")

# Compare the optimization results
print("\nComparison of constrained vs. unconstrained optimization:")
if obj_value > obj_value_constrained:
    print(f"Unconstrained optimization found better solution (improvement: {obj_value - obj_value_constrained:.4f})")
elif obj_value < obj_value_constrained:
    print(f"Constrained optimization found better solution (improvement: {obj_value_constrained - obj_value:.4f})")
else:
    print("Both methods found the same solution.")

# Save the model
predictor.save_model('electron_optics_model.pth')
print("\nModel saved to 'electron_optics_model.pth'")
"""
