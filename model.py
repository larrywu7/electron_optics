import os
import sys
import math
from typing import Tuple, Optional, Callable, Union
import torch
from torch import nn, optim
from torch.quasirandom import SobolEngine
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import importlib
import matplotlib.pyplot as plt

ROOT_DIR = (os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
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
        hidden_dims: list = [32, 64, 128, 64, 32],
        leak: float = 0.0,
        dropout: float = 0.2,
    ):
        super(ElectronOpticsModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.leak = leak
        # Build the network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(leak),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (vector output)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class _ResidualBlock(nn.Module):
    def __init__(self, dim, leak=0.2, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=14, output_dim=12, hidden_dims=64, leak=0.2, dropout=0.2, num_blocks=3):
        super().__init__()
        self.dropout = dropout
        self.leak = leak
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.LeakyReLU(leak)
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dims, leak=leak, dropout=dropout) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dims, output_dim)
        
    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)

class ElectronOpticsPredictor:
    """Main class for training and using the electron optics prediction model"""

    def __init__(
        self,
        input_dim: int=1,
        output_dim: int = 1,
        device: Optional[str] = None,
        leak: float = 0.0,
        dropout: float = 0.2,
        model=ElectronOpticsModel,
        **kwargs
    ):
        device = self.get_device() if device is None else device
        self.device = torch.device(device)
        self.dropout = dropout
        if model==ElectronOpticsModel:
            self.model = model(input_dim, output_dim, leak=leak, dropout=dropout, **kwargs).to(
                self.device
            )
        else:
            self.model = model(input_dim, output_dim, leak=leak, dropout=dropout, **kwargs).to(self.device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler_voltages = None
        self.scaler_values = None
        self.train_ds = None
        self.validation_ds = None
        self.train_losses = []
        self.val_losses = []
        self.optimization_iter = 0

        print(f"Using device: {self.device}")

    def _normalize_data(
        self, data: np.ndarray, scaler: Optional[dict] = None, fit: bool = False
    ):
        """Normalize data using min-max scaling"""
        if fit or (scaler is None):
            scaler = {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}

        
        if isinstance(data, torch.Tensor):
            normalized = (
                data
                - torch.tensor(scaler["mean"], device=self.device, dtype=torch.float32)
            ) / torch.tensor(scaler["std"], device=self.device, dtype=torch.float32)
        else:
            normalized = (data - scaler["mean"]) / scaler["std"]
        return normalized, scaler

    def _denormalize_values(self, normalized_values: np.ndarray):
        """Denormalize values back to original scale"""
        if self.scaler_values is None:
            return normalized_values
        if isinstance(normalized_values, torch.Tensor):
            denormalized = normalized_values * (
                torch.tensor(
                    (self.scaler_values["std"]),
                    device=self.device,
                    dtype=torch.float32,
                )
            ) + torch.tensor(
                self.scaler_values["mean"], device=self.device, dtype=torch.float32
            )
        else:
            denormalized = (
                normalized_values
                * (self.scaler_values["std"])
                + self.scaler_values["mean"]
            )

        return denormalized

    def _denormalize_voltages(self, normalized_voltages: np.ndarray):
        """Denormalize voltages back to original scale"""
        if self.scaler_voltages is None:
            return normalized_voltages


        return normalized_voltages * self.scaler_voltages["std"] + self.scaler_voltages["mean"]

    def train(
        self,
        voltages: np.ndarray,
        values: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        tolerance: float = 1e-6,
        patience: int = 100,
        validation_split: float = 0.2,
        verbose: bool = True,
        checkpoint_name: str = "best_model.pth",
        make_plot: bool = True,
        early_stopping: bool = True,
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
        self.train_ds = ElectronOpticsDataset(
            voltages[train_idx], values[train_idx]
        )
        self.val_ds = ElectronOpticsDataset(
            voltages[val_idx], values[val_idx]
        )
        self.train_ds_norm = train_dataset
        self.val_ds_norm = val_dataset

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Try OneCycleLR instead of ReduceLROnPlateau
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=len(train_loader)
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
                outputs = self.model(batch_voltages)
                loss: torch.Tensor = criterion(outputs, batch_values)
                optimizer.zero_grad()
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
            self.train_losses = train_losses
            self.val_losses = val_losses
           
            if val_loss < best_val_loss-tolerance:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                # J: Why save here without self.save?
                self.save_model(checkpoint_name)
            # Early stopping
            elif val_loss >= best_val_loss - tolerance and early_stopping:
                patience_counter += 1
                if patience_counter > patience:  # Early stopping patience
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )
        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_name)['model_state_dict'])

        if verbose:
            print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
            if make_plot:
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
        device: Optional[str] = None  # ← return NumPy by default for inference
    ):
        """
        If `require_grad=True`, returns a **torch.Tensor** on the same device
        so that gradients can flow.  Otherwise returns NumPy (unless you set
        `as_numpy=False`).
        """
        self.model.eval()

        # -------- 1. Convert input to a tensor on the right device ----------
        if device is None:
            device = self.device
        if isinstance(voltages, np.ndarray):
            voltages_tensor = torch.as_tensor(
                voltages, dtype=torch.float32, device=device
            )
        else:
            voltages_tensor = voltages.to(device, dtype=torch.float32)

        voltages_tensor = voltages_tensor.flatten()

        # -------- 2. Normalise (pure-torch math so graph is intact) ---------
        vmean = torch.as_tensor(
            self.scaler_voltages["mean"], dtype=torch.float32, device=device
        )
        vstd = torch.as_tensor(
            self.scaler_voltages["std"], dtype=torch.float32, device=device
        )
        # Add small epsilon to prevent division by zero
        vstd = torch.clamp(vstd, min=1e-8)
        voltages_norm = (voltages_tensor - vmean) / vstd

        voltages_norm.requires_grad_(require_grad)
        self.model.to(device)
        # -------- 3. Forward pass ------------------------------------------
        with torch.set_grad_enabled(require_grad):
            preds_norm: torch.Tensor = self.model(
                voltages_norm.unsqueeze(0)
            )  # (1, n_out)

            # denormalise **in torch**
            if self.scaler_values is not None:
                smean = torch.as_tensor(
                    self.scaler_values["mean"], dtype=torch.float32, device=device
                )
                sstd = torch.as_tensor(
                    self.scaler_values["std"], dtype=torch.float32, device=device
                )
                preds = preds_norm * sstd + smean
            else:
                preds = preds_norm

        preds = preds.squeeze(0)  # shape (n_out,)

        # -------- 4. Decide return type ------------------------------------
        
        return preds  # torch.Tensor with grad_fn (if any)
        

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {   "model_name": self.model.__class__.__name__,
                "model_state_dict": self.model.state_dict(),
                "scaler_voltages": self.scaler_voltages,
                "scaler_values": self.scaler_values,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dims": self.model.hidden_dims,
                "num_blocks": self.model.num_blocks,
                "leak": self.model.leak,
                "dropout": self.model.dropout,
                "train_ds": self.train_ds,
                "validation_ds": self.validation_ds,
                "train_ds_norm": self.train_ds_norm,
                "val_ds_norm": self.val_ds_norm,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
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
        mod=importlib.import_module("model")
        # Recreate model with proper dimensions
        predictor = cls(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint.get("output_dim", 1),
            device=device,
            model=getattr(mod, checkpoint.get("model_name")),
            leak=checkpoint.get("leak"),
            dropout=checkpoint.get("dropout"),
            hidden_dims=checkpoint.get("hidden_dims"),
            num_blocks=checkpoint.get("num_blocks"),
            
            
        )
        # Load state and scalers

        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.scaler_voltages = checkpoint["scaler_voltages"]
        predictor.scaler_values   = checkpoint["scaler_values"]
        predictor.input_dim       = checkpoint["input_dim"]
        predictor.output_dim      = checkpoint["output_dim"]
        predictor.train_ds        = checkpoint["train_ds"]
        predictor.validation_ds   = checkpoint["validation_ds"]
        predictor.train_ds_norm   = checkpoint["train_ds_norm"]
        predictor.val_ds_norm     = checkpoint["val_ds_norm"]
        predictor.train_losses    = checkpoint.get("train_losses", [])
        predictor.val_losses      = checkpoint.get("val_losses", [])
        
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
    hyperparameters: dict = {'strength': 0, 'min': float('-inf'), 'max': float('inf'), 's': 100000},
    device= None,
    clamp: bool = False,
    initial_voltages: Optional[np.ndarray] = None,
    seed: Optional[int] = 0,
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
    device = predictors[0].device if device is None else device

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
    if initial_voltages and len(initial_voltages)==1:
            initial_voltages =[initial_voltages]
    sob=None
    for restart in range(random_restarts):
        losses = []
        metric_values = []
        regularizer_values = []
        rhos = []
        # Initialize voltages
        
        if voltage_bounds:
            if initial_voltages:
                voltages = initial_voltages.pop()
                voltages = np.array(voltages)
                 # only use initial voltages for first restart
            else:
                if sob is None:
                    sob = SobolEngine(14, scramble=True, seed=seed)
                voltages = sob.draw(1).squeeze(0).numpy()
                voltages = voltage_bounds[0] + (voltage_bounds[1]-voltage_bounds[0]) * voltages
                # voltages = np.random.uniform(voltage_bounds[0], voltage_bounds[1],size=predictors[0].input_dim)
        else:
            if initial_voltages:
                voltages = initial_voltages.pop()
                voltages = np.array(voltages)
                 # only use initial voltages for first restart
            else:
                if sob is None:
                    sob = SobolEngine(predictors[0].input_dim, scramble=True, seed=seed)
                voltages = sob.draw(1).squeeze(0).numpy()
                voltages = np.min(predictors[0].train_ds.voltages.numpy(), axis=0) + (np.max(predictors[0].train_ds.voltages.numpy(), axis=0)-np.min(predictors[0].train_ds.voltages.numpy(), axis=0)) * voltages 
            #     voltages = np.random.uniform(

            #     np.min(predictors[0].train_ds.voltages.numpy(), axis=0),
            #     np.max(predictors[0].train_ds.voltages.numpy(), axis=0),
            #     size=predictors[0].input_dim,
            # )

        # Normalize initial voltages

        voltages = voltages.flatten()

        # Convert to tensor and require gradients
        voltages_tensor = torch.FloatTensor(voltages).to(device)
        voltages_tensor.requires_grad_(True)

        optimizer = optim.Adam([voltages_tensor], lr=learning_rate)

        for iteration in range(n_iterations):
            

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

            strength, a, b = hyperparameters['strength'], hyperparameters['min'], hyperparameters['max']
            u = tanh_transform(voltages_tensor, a, b)  # apply sigmoid transform so that the voltages stay within [a,b]

            # Add epsilon to prevent log(0) and ensure numerical stability
            eps = 1e-8
            u_safe = torch.clamp(u, min=a + eps, max=b - eps)
            regularizing_term = -strength * (torch.log((u_safe-a)/(b-a) + eps) + torch.log((b-u_safe)/(b-a) + eps)).sum()

            # Compute loss 

            metric_value = (objective_func(predictions)).to(device)
            loss =  metric_value + regularizing_term
            
            losses.append((loss).item())
            metric_values.append(metric_value.item())
            regularizer_values.append(regularizing_term.item())
            log_every = max(1, n_iterations // 10)
            if iteration % log_every == 0:
                gt = torch.autograd.grad(metric_value, voltages_tensor, retain_graph=True, allow_unused=True)
                gr = torch.autograd.grad(regularizing_term, voltages_tensor, retain_graph=True, allow_unused=True)
                rho = (l2(gr) / (l2(gt) + 1e-12)).item()
                rhos.append(rho)
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients and clip them
            if torch.isnan(loss):
                print(f"Warning: Loss is NaN at iteration {iteration}")
                break
                
            # Clip gradients to prevent exploding gradients
            if voltages_tensor.grad is not None:
                clip_grad_norm_(voltages_tensor, max_norm=1.0)
            
            # Check for NaN in voltages_tensor before step
            if torch.isnan(voltages_tensor).any():
                print(f"Warning: voltages_tensor contains NaN before optimizer step at iteration {iteration}")
                break
                
            optimizer.step()
            
            # Check for NaN in voltages_tensor after step
            if torch.isnan(voltages_tensor).any():
                print(f"Warning: voltages_tensor contains NaN after optimizer step at iteration {iteration}")
                # Reset voltages_tensor to last good values or reinitialize
                break
            

            # Optionally constrain to training range
            if clamp:
                with torch.no_grad():
                    if voltage_bounds is not None:
                        voltages_tensor.clamp_(
                            torch.tensor(voltage_bounds[0], device=device, dtype=torch.float32),
                            torch.tensor(voltage_bounds[1], device=device, dtype=torch.float32)
                        )
                    else:
                        voltages_tensor.clamp_(np.min(predictors[0].train_ds.voltages.numpy()), np.max(predictors[0].train_ds.voltages.numpy()))

        fig,ax = plt.subplots(1,2,figsize=(8,6))
        ax[0].plot(losses, label=f"loss (Restart {restart+1})")
        ax[0].plot(metric_values, label=f"metric (Restart {restart+1})", linestyle='dashed', color=ax[0].get_lines()[0].get_color())
        ax[0].plot(regularizer_values, label=f"regularizer (Restart {restart+1})", linestyle='dotted', color=ax[0].get_lines()[0].get_color())
        ax[0].legend()
        ax[0].set_yscale("log")
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("Loss / Metric / Regularizer")
        ax[0].set_title(f"Voltage Optimization Progress (Restart {restart+1})")
        if len(rhos) > 0:
            ax[1].plot(np.arange(0, n_iterations, max(1, n_iterations // 10)), rhos, label=f"rho (Restart {restart+1})", color=ax[0].get_lines()[0].get_color())
            ax[1].set_xlabel("iteration")
            ax[1].set_ylabel("rho")
            ax[1].set_title("Gradient Ratio (Regularizer / Metric)")
        plt.tight_layout()
        # Get final result
        with torch.no_grad():
            final_predictions = torch.empty((0,), dtype=torch.float32, device=device)
            for predictor in predictors:
                final_prediction = predictor.predict(
                    voltages_tensor.unsqueeze(0), require_grad=False, device=device
                )
                final_predictions = torch.cat(
                    (final_predictions, final_prediction), dim=0
                )
            final_predictions = final_predictions.to(device)

            final_objective = (objective_func(final_predictions)).item()

            if final_objective < best_objective:
                best_objective = final_objective
                best_voltages = voltages_tensor.cpu().numpy()
                best_values = final_predictions
 
    return (
        best_voltages,
        best_values,
        best_objective,
    )  # best_values is best predicted output_values and best_objective is best metric value.

def optimize_voltages_gradnorm(
    predictors: list[ElectronOpticsPredictor],
    objective_function: Optional[Callable] = None,
    value_index: int = 0,
    weights: list = None,
    n_iterations: int = 1000,
    learning_rate: float = 1e-5,
    learning_rate_weights: float = 5e-4,
    random_restarts: int = 5,
    voltage_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    hyperparameters: dict = {'strength': 0, 'min': float('-inf'), 'max': float('inf'), 's': 100000, 'alpha': 0.12},
    seed: Optional[int] = 0,
    device='cpu',
    clamp: bool = False,
    initial_voltages: Optional[np.ndarray] = None,
    figsize=(8,20)
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
    fig,ax = plt.subplots(random_restarts,2,figsize=figsize)
    ax = np.atleast_2d(ax)
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
    if initial_voltages and len(initial_voltages)==1:
        initial_voltages = [initial_voltages]
    restart_results = []
    sob=None
    log_weights_restarts=[]
    log_loss_restarts=[]
    for restart in range(random_restarts):
        losses = []
        metric_values = []
        regularizer_values = []
        rhos = []
        # Initialize voltages
       
        if voltage_bounds:
            if initial_voltages:
                voltages = initial_voltages.pop()
                voltages = np.array(voltages)
                 # only use initial voltages for first restart
            else:
                if sob is None:
                    sob = SobolEngine(14, scramble=True, seed=seed)
                voltages = sob.draw(1).squeeze(0).numpy()
                voltages = voltage_bounds[0] + (voltage_bounds[1]-voltage_bounds[0]) * voltages
                # voltages = np.random.uniform(voltage_bounds[0], voltage_bounds[1],size=predictors[0].input_dim)
        else:
            if initial_voltages:
                voltages = initial_voltages.pop()
                voltages = np.array(voltages)
                 # only use initial voltages for first restart
            else: 
                if restart==0:
                    sob = SobolEngine(predictors[0].input_dim, scramble=True, seed=seed)
                voltages = sob.draw(1).squeeze(0).numpy()
                voltages = np.min(predictors[0].train_ds.voltages.numpy(), axis=0) + (np.max(predictors[0].train_ds.voltages.numpy(), axis=0)-np.min(predictors[0].train_ds.voltages.numpy(), axis=0)) * voltages 

        # Normalize initial voltages

        voltages = voltages.flatten()

        # Convert to tensor and require gradients
        voltages_tensor = torch.FloatTensor(voltages).to(device)
        voltages_tensor.requires_grad_(True)
        

        optimizer = optim.Adam([voltages_tensor], lr=learning_rate)



        log_weights = []
        log_loss = []
        for iteration in range(n_iterations):
            

            # Predict values
            predictions = torch.empty(0, dtype=torch.float32, device=device)

            for predictor in predictors:
                prediction = predictor.predict(
                    voltages_tensor.unsqueeze(0)[:, : predictor.input_dim],
                    require_grad=True,device=device
                )
                prediction = torch.Tensor(prediction).squeeze(0).to(device)
                predictions = torch.cat((predictions, prediction), dim=0)

            predictions = predictions.to(device)

            if hyperparameters['strength'] != 0:
                strength, a, b = hyperparameters['strength'], hyperparameters['min'], hyperparameters['max']
                u = tanh_transform(voltages_tensor, a, b, device=device)  # apply sigmoid transform so that the voltages stay within [a,b]
                if (u.isnan()).any():
                   print("NaN encountered in voltage transform")
                regularizing_term = -strength * (torch.log((u-a)/(b-a))+torch.log((b-u)/(b-a))).sum()
                if regularizing_term==np.inf:
                    print("Inf encountered in regularizing term")
            # Compute loss 

            metric_values = objective_func(predictions,device=device)
            loss =torch.cat((*metric_values, regularizing_term.unsqueeze(0)), dim=0) if hyperparameters['strength'] != 0 else torch.cat(metric_values, dim=0)
            if iteration == 0:
                weights = torch.ones_like(loss)
                weights = torch.nn.Parameter(weights)
                T=weights.sum().detach()
                optimizer_weights = torch.optim.Adam([weights], lr=learning_rate_weights)
                l0=loss.detach()
            weighted_loss = weights @ loss
            optimizer.zero_grad()
            weighted_loss.backward(retain_graph=True)

            gw=[]
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i] * loss[i], voltages_tensor, retain_graph=True, create_graph=True)[0]

                gw.append(torch.norm(dl))
                
            gw = torch.stack(gw)

            loss_ratio = loss.detach()/l0
            
            rt=loss_ratio / loss_ratio.mean()
            
            gw_avg =gw.mean().detach()
            alpha = hyperparameters['alpha']
            constant = (gw_avg * rt **alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            optimizer_weights.zero_grad()
            gradnorm_loss.backward()
            log_weights.append(weights.detach().cpu().numpy().copy())
            log_loss.append(loss_ratio.detach().cpu().numpy().copy())
            optimizer.step()
            optimizer_weights.step()
            if clamp:
                with torch.no_grad():
                    if voltage_bounds is not None:
                        voltages_tensor.clamp_(
                            torch.tensor(voltage_bounds[0], device=device, dtype=torch.float32),
                            torch.tensor(voltage_bounds[1], device=device, dtype=torch.float32)
                        )
                    else:
                        voltages_tensor.clamp_(np.min(predictors[0].train_ds.voltages.numpy()), np.max(predictors[0].train_ds.voltages.numpy()))





            weights =(weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            
            optimizer_weights = torch.optim.Adam([weights], lr=0.01)

            


        log_weights = np.array(log_weights).T
        log_weights_restarts.append(log_weights)
        log_loss = np.array(log_loss).T
        log_loss_restarts.append(log_loss)

        for i in range(log_weights.shape[0]):
            ax[restart, 0].plot(log_weights[i], label=f"weight {i} (Restart {restart+1})")
            ax[restart, 1].plot(log_loss[i], label=f"loss (Restart {restart+1}, part {i})")
        ax[restart, 0].legend()
        ax[restart, 1].legend()
        ax[restart, 0].set_yscale("log")
        ax[restart, 1].set_yscale("log")
        ax[restart, 0].set_xlabel("iteration")
        ax[restart, 1].set_xlabel("iteration")
        ax[restart, 0].set_ylabel("Weights")
        ax[restart, 1].set_ylabel("Loss")
        ax[restart, 0].set_title(f"Voltage Optimization Progress (Restart {restart+1})")
        ax[restart, 1].set_title(f"Voltage Optimization Progress (Restart {restart+1})")
        plt.tight_layout()
        # Get final result
        with torch.no_grad():
            final_predictions = torch.empty((0,), dtype=torch.float32, device=device)
            for predictor in predictors:
                final_prediction = predictor.predict(
                    voltages_tensor.unsqueeze(0)[:, : predictor.input_dim], require_grad=False, device=device
                )
                final_predictions = torch.cat(
                    (final_predictions, final_prediction), dim=0
                )
            final_predictions = final_predictions.to(device)

            final_objective = (torch.stack(objective_func(final_predictions)).sum()).item()
            restart_result={"voltages": voltages_tensor, "objective": final_objective}
            restart_results.append(restart_result)
            if final_objective < best_objective:
                best_objective = final_objective
                best_voltages = voltages_tensor.cpu().numpy()
                best_values = final_predictions
                best_restart = restart
    print(f'Restart {best_restart+1}, best objective: {best_objective}')
    return (
        best_voltages,
        best_values,
        best_objective,
    ), restart_results, (np.stack(log_weights_restarts), np.stack(log_loss_restarts))  # best_values is best predicted output_values and best_objective is best metric value.




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
def angle_resolved_aper0(aper0_map: torch.Tensor,device=torch.device("cpu")) -> torch.Tensor:
    """aper0_map shape: (6,) tensor on *any* device"""
    aper0_map = aper0_map.to(device)  # move to same GPU/CPU
    APER_0_D = torch.tensor(0.5, device=device, dtype=dtype)
    solid_angle_scaling = torch.tensor(
        math.sin(math.radians(5 / 2)) / math.sin(math.radians(60 / 2)),
        device=device, dtype=dtype,
    )
    aberr = (
        aper0_map[0] ** 2
        + aper0_map[2] ** 2
        + (aper0_map[3] * solid_angle_scaling) ** 2
        + (aper0_map[4] * solid_angle_scaling**2) ** 2
        + (aper0_map[5] * solid_angle_scaling**3) ** 2
    )

    return aberr + (torch.abs(aper0_map[1] * solid_angle_scaling) - APER_0_D / 2) ** 2


def spatial_resolved_detector(det_map: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
    """det_map shape: (≥6,) tensor on *any* device"""
    det_map = det_map.to(device)
    DET_D = torch.tensor(25.0, device=device, dtype=dtype)
    solid_angle_scaling = torch.tensor(
        math.sin(math.radians(5 / 2)) / math.sin(math.radians(60 / 2)),
        device=device, dtype=dtype,
    )
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
    return torch.stack([angle_resolved_aper0(output[:6]) * 1000 + spatial_resolved_detector(
        output[6:]
    )])
def metric_split(output: torch.Tensor,device= None) -> Tuple[torch.Tensor, torch.Tensor]:
    """`output` is the concatenated predictor output tensor."""
    device = output.device if device is None else device
    return angle_resolved_aper0(output[:6],device=device).unsqueeze(0), spatial_resolved_detector(
        output[6:], device=device
    ).unsqueeze(0)
def objective(output: torch.Tensor, device: torch.device = None):
    device = torch.device("cpu") if device is None else device
    output.to(device)
    angle_scaling = 1000
    angle_aberrations = (
        output[0] ** 2
        + output[2] ** 2
        + (output[3] * solid_angle_scaling) ** 2
        + (output[4] * (solid_angle_scaling**2)) ** 2
        + (output[5] * (solid_angle_scaling**3)) ** 2
    )
    spatial_aberrations = (
        (output[1 + 6] * solid_angle_scaling) ** 2
        + output[2 + 6] ** 2
        + (output[3 + 6] * solid_angle_scaling) ** 2
        + (output[4 + 6] * (solid_angle_scaling**2)) ** 2
        + (output[5 + 6] * (solid_angle_scaling**3)) ** 2
    )
    return (
        (torch.abs(output[1]) * solid_angle_scaling - APER_0_D / 2) ** 2 * angle_scaling
        + (torch.abs(output[6]) - DET_D / 2) ** 2
        + angle_aberrations * angle_scaling
        + spatial_aberrations
    )