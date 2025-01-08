"""POTNet module."""

import os
import pickle
import warnings
from typing import Dict, Optional, Tuple, Union
import re

import numpy as np
import ot
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .data_transformer import DataTransformer, DiscreteDataTransformer



def dense_block(
    in_features: int,
    out_features: int,
    activation: nn.Module = nn.ReLU(inplace=True),
    use_batchnorm: bool = True,
    use_dropout: bool = False,
    dropout_rate: float = 0.1
) -> nn.Sequential:
    """
    Creates a small block of layers: Linear -> (BatchNorm) -> (Dropout) -> Activation.
    The final layer in the network should typically skip BN, Dropout, and final activation.
    """
    layers = [nn.Linear(in_features, out_features)]
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_features))
    if use_dropout:
        layers.append(nn.Dropout(p=dropout_rate))
    layers.append(activation)
    return nn.Sequential(*layers)


class POTNetGenerator(nn.Module):
    """
    POTNetGenerator with skip connections between input and hidden layers.
    Optional features:
      - Batch normalization
      - Dropout
      - Skip connections
      - Discrete column transformation
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        num_disc_cols: int,
        architecture: List[int],
        disc_col_category_idx: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        dropout: bool = True,
        dropout_rate: float = 0.1,
        skip: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_disc_cols = num_disc_cols
        self.disc_col_category_idx = disc_col_category_idx
        self.skip = skip

        if architecture[0] != embedding_dim:
            raise ValueError("First layer dimension in 'architecture' must match 'embedding_dim'.")
        if architecture[-1] != output_dim:
            raise ValueError("Last layer dimension in 'architecture' must match 'output_dim'.")

        self.hidden_layers = nn.ModuleList()
        num_layers = len(architecture)

        # Build hidden layers 
        for i in range(num_layers - 2):
            in_dim = architecture[i]
            # Add skip connection dimension if enabled and not the first layer
            if skip and i > 0:
                in_dim += architecture[0]
            out_dim = architecture[i + 1]

            block = dense_block(
                in_features=in_dim,
                out_features=out_dim,
                activation=activation,
                use_batchnorm=batch_norm,
                use_dropout=dropout,
                dropout_rate=dropout_rate
            )
            self.hidden_layers.append(block)

        # Final layer: no BN/Dropout/Activation
        final_in_dim = architecture[-2]
        # if there was at least one hidden layer
        if skip and (num_layers - 2) > 0:  
            final_in_dim += architecture[0]
        self.final_layer = nn.Linear(final_in_dim, architecture[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = x.view(-1, self.embedding_dim).to(x.device)
        skip_input = x

        # Pass through hidden layers
        for i, block in enumerate(self.hidden_layers):
            if self.skip and i > 0:
                x = torch.cat([x, skip_input], dim=-1)
            x = block(x)

        # Final linear layer (no activation/BN/Dropout)
        if self.skip and len(self.hidden_layers) > 0:
            x = torch.cat([x, skip_input], dim=-1)
        x = self.final_layer(x)

        # Softmax for discrete columns
        if self.num_disc_cols > 0 and self.disc_col_category_idx is not None:
            prev_c = 0
            for c in range(len(self.disc_col_category_idx)):
                cur_idx = self.disc_col_category_idx[c]
                prev_idx = self.disc_col_category_idx[prev_c]
                x[:, prev_idx:cur_idx] = F.softmax(x[:, prev_idx:cur_idx], dim=1)
                prev_c = c

        return x



class POTNetCondGenerator(nn.Module):
    """
    POTNetCondGenerator module for POTNet with skip connections.
    Accepts a noise vector and a conditioning vector, merges them as input, 
    and optionally skips the input across hidden layers.

    Key ideas:
      - Hidden layers each built with 'dense_block' 
        (Linear -> BN -> Dropout -> Activation).
      - A final linear layer (no BN, Dropout, or Activation).
      - Optional skip connections from the concatenated input to each hidden layer.
      - Optional softmax for discrete columns.
    """

    def __init__(
        self,
        embedding_dim: int,
        cond_dim: int,
        output_dim: int,
        num_disc_cols: int,
        architecture: List[int],
        disc_col_category_idx: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        dropout: bool = False,
        dropout_rate: float = 0.1,
        skip: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        self.input_dim = embedding_dim + cond_dim
        self.output_dim = output_dim
        self.num_disc_cols = num_disc_cols
        self.disc_col_category_idx = disc_col_category_idx
        self.skip = skip

        if architecture[0] != self.input_dim:
            raise ValueError("First layer dimension in 'architecture' must match 'embedding_dim + cond_dim'.")
        if architecture[-1] != output_dim:
            raise ValueError("Last layer dimension in 'architecture' must match 'output_dim'.")

        self.hidden_layers = nn.ModuleList()
        self._build_network(
            architecture=architecture,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_rate=dropout_rate
        )

    def _build_network(
        self,
        architecture: List[int],
        activation: nn.Module,
        batch_norm: bool,
        dropout: bool,
        dropout_rate: float
    ) -> None:
        """
        Builds hidden layers (with dense_block) and a final linear layer without BN/Dropout/Activation.
        """
        num_layers = len(architecture)

        # Build hidden layers: all but the last
        for i in range(num_layers - 2):
            in_dim = architecture[i]
            if self.skip and i > 0:
                in_dim += architecture[0]  # Add skip dimension only beyond first hidden layer

            out_dim = architecture[i + 1]
            block = dense_block(
                in_features=in_dim,
                out_features=out_dim,
                activation=activation,
                use_batchnorm=batch_norm,
                use_dropout=dropout,
                dropout_rate=dropout_rate
            )
            self.hidden_layers.append(block)

        # Final layer (no BN/Dropout/Activation)
        final_in_dim = architecture[-2]
        if self.skip and (num_layers - 2) > 0:  # If there's at least one hidden layer
            final_in_dim += architecture[0]
        self.final_layer = nn.Linear(final_in_dim, architecture[-1])

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the conditional generator.
        Final softmax for discrete columns.
        """
        x = torch.cat((noise, cond), dim=1).view(-1, self.input_dim)
        skip_input = x  # For skip connections

        # Pass through hidden layers
        for i, block in enumerate(self.hidden_layers):
            if self.skip and i > 0:
                x = torch.cat([x, skip_input], dim=-1)
            x = block(x)

        # Final layer
        if self.skip and len(self.hidden_layers) > 0:
            x = torch.cat([x, skip_input], dim=-1)
        x = self.final_layer(x)

        # Softmax for discrete columns
        if self.num_disc_cols > 0 and self.disc_col_category_idx is not None:
            prev_c = 0
            for c in range(len(self.disc_col_category_idx)):
                cur_idx = self.disc_col_category_idx[c]
                prev_idx = self.disc_col_category_idx[prev_c]
                x[:, prev_idx:cur_idx] = F.softmax(x[:, prev_idx:cur_idx], dim=1)
                prev_c = c

        return x


class POTNetDataset(Dataset):
    """
    Dataset class for POTNet.
    Holds data and (optionally) sample weights.
    """

    def __init__(self, data, weights=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(data), dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.weights[idx]


class POTNetConditionalDataset(Dataset):
    """
    Dataset class for conditional POTNet.
    Retains both conditioning and target data, plus optional weights.
    """

    def __init__(self, conditioning_data, data, weights=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.conditioning_data = torch.tensor(conditioning_data, dtype=torch.float32)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(data), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.conditioning_data[idx], self.data[idx], self.weights[idx]


class POTNet:
    """
    Penalized Optimal Transport Network (POTNet) class.
    This class implements a generative model using a penalized optimal transport approach. It supports both
    unconditional and conditional generation. The user can specify transformations for discrete and continuous
    features, sample weights, and checkpointing for training continuation.
    """

    def __init__(
        self,
        embedding_dim=None,
        architecture=(256, 256),
        conditional=False,
        categorical_cols=None,
        numeric_output_data_type='continuous',
        standardize=True,
        p=1.0,
        mp_lambdas=1.,
        activation=nn.ReLU(inplace=True),
        batch_norm=True,
        dropout=True,
        dropout_rate=0.1,
        skip=True,
        mp_decay=0.996,
        epochs=300,
        batch_size=256,
        lr=1e-3,
        betas=(0.9, 0.999),
        verbose=False,
        save_checkpoint=False,
        resume_checkpoint=None,
        checkpoint_path='./checkpoints',
        checkpoint_name='potnet_ckpt_iter',
        checkpoint_epoch=10,
        overwrite_checkpoint=False,
        print_loss=False,
        print_loss_iter=10
    ):
        """
        Initializes a model with the specified hyperparameters.

        Args:
            embedding_dim: Dimension of the noise vector. Defaults to None.
            architecture: List of hidden layer dimensions. Defaults to (256, 256).
            conditional: If True, the model is conditional. Defaults to False.
            categorical_cols: Discrete columns in data (excluding conditional data) Defaults to None.
            numeric_output_data_type: 'integer' or 'continuous', or dict of type and associated columns {'integer': [col1, col2], 'continuous': [col3, col4]}
            standardize: If True, standardizes continuous data. Defaults to True.
            p: Power for the Wasserstein distance. Defaults to 1.0.
            mp_lambdas: Marginal penalty for each column. Defaults to 1.
            activation: Activation function for hidden layers. Defaults to ReLU.
            batch_norm: If True, uses batch normalization. Defaults to True.
            dropout: If True, uses dropout. Defaults to True.
            dropout_rate: Dropout rate. Defaults to 0.1.
            skip: If True, uses skip connections. Defaults to True.
            mp_decay: Decay factor for marginal penalties. Defaults to 0.996.
            epochs: Number of training epochs. Defaults to 300.
            batch_size: Training batch size. Defaults to 256.
            lr: Learning rate. Defaults to 1e-3.
            betas: Adam optimizer betas. Defaults to (0.9, 0.999).
            verbose: If True, prints training progress. Defaults to False.
            save_checkpoint: If True, saves a checkpoint every 'checkpoint_epoch' epochs. Defaults to False.
            resume_checkpoint: Path to a checkpoint for resuming training. Defaults to None.
            checkpoint_path: Directory for saving checkpoints. Defaults to './checkpoints'.
            checkpoint_name: Base name for checkpoint files. Defaults to 'potnet_ckpt_iter'.
            checkpoint_epoch: Epoch interval for saving checkpoints. Defaults to 10.
            overwrite_checkpoint: If True, overwrites existing checkpoints. Defaults to False.
            print_loss: If True, prints loss during training. Defaults to False.
            print_loss_iter: Iteration interval for printing loss. Defaults to 10.
        """

        self._embedding_dim = embedding_dim
        self._output_dim = None
        self._conditional = conditional
        self._cond_dim = None
        self._categorical_cols = categorical_cols
        self._disc_data_transformer = DiscreteDataTransformer(categorical_cols)
        if self._categorical_cols is not None:
            self._num_disc_cols = len(categorical_cols)
        else:
            self._num_disc_cols = 0
        self._disc_col_category_idx = None
        self._p = p
        self._resume_checkpoint = resume_checkpoint
        self._save_checkpoint = save_checkpoint
        self._checkpoint_path = checkpoint_path
        self._checkpoint_name = checkpoint_name
        self._checkpoint_epoch = checkpoint_epoch
        self._overwrite_checkpoint = overwrite_checkpoint
        self._print_loss = print_loss
        self._print_loss_iter = print_loss_iter


        if isinstance(activation, str):
            chosen = activation.lower()
            acts = {
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'leakyrelu': nn.LeakyReLU(),
                'elu': nn.ELU(),
                'swish': nn.SiLU(),
                'gelu': nn.GELU()
            }
            self._activation = acts.get(chosen, nn.SiLU())
        else:
            self._activation = activation

        self._dropout = dropout
        self._dropout_rate = dropout_rate
        self._skip = skip
        self._batch_norm = batch_norm
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._betas = betas
        self._verbose = verbose
        self._standardize = standardize
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._input_architecture = architecture
        self._architecture = None
        self._mp_lambdas = mp_lambdas
        self._mp_decay = mp_decay
        self._cur_epoch = 0

        assert self._mp_lambdas is not None, "mp_lambdas must be provided."
        assert self._mp_lambdas >= 0, "mp_lambdas must be nonnegative."


        if isinstance(numeric_output_data_type, dict):
            if not all(k in ['integer', 'continuous'] for k in numeric_output_data_type):
                raise ValueError("Invalid numeric output data type keys (must be 'integer' or 'continuous').")
        else:
            if numeric_output_data_type not in ['integer', 'continuous']:
                raise ValueError("numeric_output_data_type must be 'integer' or 'continuous'.")
        self.numeric_output_data_type = numeric_output_data_type

        if self._num_disc_cols > 0:
            self._transformer = DataTransformer(data_type=self.numeric_output_data_type, transformation='min_max')
        else:
            if self._standardize:
                self._transformer = DataTransformer(data_type=self.numeric_output_data_type, transformation='standardize')
            else:
                self._transformer = DataTransformer(data_type=self.numeric_output_data_type, transformation=None)

        self._data_sampler = None
        self._generator = None
        self._optimizer = None

        if self._resume_checkpoint:
            self.load_checkpoint(self._resume_checkpoint)
        

    @staticmethod
    def _init_weights_normal(m):
        cname = m.__class__.__name__
        if cname.find("Linear") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 1 / np.sqrt(10000))

    @staticmethod
    def _check_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')
    
    @staticmethod
    def _check_existence_of_checkpoints(overwrite_checkpoint,
                                        start_epoch,
                                        epochs,
                                        checkpoint_epoch,
                                        checkpoint_path,
                                        checkpoint_name):
        if not overwrite_checkpoint:
            save_iters = range(start_epoch, epochs, checkpoint_epoch)
            for iter_num in save_iters:
                ckpt_iter_name = f'{checkpoint_name}{iter_num}.pth'
                full_path = os.path.join(checkpoint_path, ckpt_iter_name)
                if os.path.exists(full_path):
                    raise FileExistsError(f"Checkpoint file already exists: {full_path}")

    def get_loader(self, data, conditioning_data=None, weights=None):
        """
        Constructs and returns a DataLoader object for training or inference.
        """
        if isinstance(self._mp_lambdas, float):
            self._mp_lambdas = np.ones(data.shape[1]) * self._mp_lambdas
        if self._mp_lambdas is not None and self._mp_lambdas is not float and len(self._mp_lambdas) != data.shape[1]:
            raise ValueError("mp_lambdas must match data dimension length.")

        if self._categorical_cols is not None:
            disc_data, num_data, self._mp_lambdas, num_disc_cols, disc_col_category_idx = self._disc_data_transformer.one_hot_encode(
                data,
                self._categorical_cols,
                self._mp_lambdas
            )
            self._disc_col_category_idx = disc_col_category_idx
            self._num_disc_cols = num_disc_cols

        if self._num_disc_cols > 0:
            if num_data is not None:
                num_data = self._transformer.standardize(num_data, column_names=self._disc_data_transformer._cont_data_columns)
        else:
            if isinstance(data, pd.DataFrame):
                data = data.astype(float).to_numpy()
            data = self._transformer.standardize(data)

        if self._conditional:
            if self._num_disc_cols > 0:
                self._cond_data_transformer = DataTransformer(data_type='continuous', transformation='min_max')
                colnames = self._disc_data_transformer._cont_data_columns
            else:
                if self._standardize:
                    self._cond_data_transformer = DataTransformer(data_type='continuous', transformation='standardize')
                else:
                    self._cond_data_transformer = DataTransformer(data_type='continuous', transformation=None)
                colnames = None
                conditioning_data = self._cond_data_transformer.standardize(conditioning_data, column_names=colnames)
            self._cond_data = conditioning_data

        if self._num_disc_cols > 0:
            if num_data is not None:
                data = np.concatenate((disc_data, num_data), axis=1)
            else:
                data = disc_data

        if self._conditional:
            dataset = POTNetConditionalDataset(conditioning_data, data, weights)
        else:
            dataset = POTNetDataset(data, weights)

        self._output_dim = data.shape[1]
        if self._embedding_dim is None:
            self._embedding_dim = self._output_dim
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

    @staticmethod
    def _contains_nan(data):
        if isinstance(data, np.ndarray):
            return np.isnan(data).any()
        elif isinstance(data, pd.DataFrame):
            return data.isna().any().any()
        else:
            raise TypeError("Input data must be a NumPy array or a Pandas DataFrame.")
        

    def fit(
        self,
        train_data,
        conditioning_data=None,
        weights=None,
        epochs=None,
        save_checkpoint=False,
        resume_checkpoint=None,
        checkpoint_path='./checkpoints',
        checkpoint_name='potnet_ckpt_iter',
        checkpoint_epoch=10,
        overwrite_checkpoint=False 
    ):
        """
        Fits the POTNet model on the provided data.

        Args:
            train_data: Training data.
            conditioning_data: Conditioning data for the conditional variant.
            weights: Importance weights for each training data. If provided, should be of same length as training data.
            epochs: Number of training epochs.
            save_checkpoint: If True, saves a checkpoint every 'checkpoint_epoch' epochs.
            resume_checkpoint: Path to a checkpoint for resuming training.
            checkpoint_path: Directory for saving checkpoints.
            checkpoint_epoch: Epoch interval for saving checkpoints.
            overwrite_checkpoint: If True, overwrites existing checkpoints.
        """
        if self._contains_nan(train_data):
            raise ValueError("Training data contains missing values.")
        if self._verbose:
            print("Fitting POTNet...")
        if self._check_interactive():
            # Jupyter notebook environment
            from tqdm.notebook import tqdm
        else:
            from tqdm.auto import tqdm

        use_weights = weights is not None
        start_epoch = 0

        if self._conditional:
            self._cond_dim = conditioning_data.shape[1]
            dataloader = self.get_loader(train_data, conditioning_data=conditioning_data, weights=weights)
        else:
            dataloader = self.get_loader(train_data, weights=weights)

        self._architecture = (
            [self._embedding_dim + self._cond_dim] + list(self._input_architecture) + [self._output_dim]
            if self._conditional
            else [self._embedding_dim] + list(self._input_architecture) + [self._output_dim]
        )
        if len(self._mp_lambdas) != self._output_dim:
            raise ValueError("mp_lambdas length must match output dimension.")
        self._mp_lambdas = torch.as_tensor(self._mp_lambdas, device=self._device, dtype=torch.float32)

        if self._conditional:
            self._cond_data = Variable(torch.tensor(self._cond_data, dtype=torch.float32)).to(self._device)

        if self._generator is None:
            # Initialize the generator
            if self._conditional:
                self._generator = POTNetCondGenerator(
                    self._embedding_dim,
                    self._cond_dim,
                    self._output_dim,
                    self._num_disc_cols,
                    self._architecture,
                    self._disc_col_category_idx,
                    self._activation,
                    self._batch_norm,
                    self._dropout,
                    self._dropout_rate,
                    self._skip
                ).to(self._device)
            else:
                self._generator = POTNetGenerator(
                    self._embedding_dim,
                    self._output_dim,
                    self._num_disc_cols,
                    self._architecture,
                    self._disc_col_category_idx,
                    self._activation,
                    self._batch_norm,
                    self._dropout,
                    self._dropout_rate,
                    self._skip
                ).to(self._device)
            self._optimizer = torch.optim.AdamW(self._generator.parameters(), lr=self._lr, betas=self._betas)

        self._save_checkpoint = save_checkpoint
        self._resume_checkpoint = resume_checkpoint
        if save_checkpoint:
            self._checkpoint_epoch = checkpoint_epoch
            self._checkpoint_path = checkpoint_path
            self._checkpoint_name = checkpoint_name
            self._overwrite_checkpoint = overwrite_checkpoint


        if resume_checkpoint:
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.load_checkpoint(resume_checkpoint)
            start_epoch = self._cur_epoch + 1
            print(f"Resumed training from epoch {start_epoch}.")
        else:
            # Initialize network weights
            self._generator.apply(self._init_weights_normal)
            self._generator.train()
        
        if epochs is not None:
            self._epochs = epochs

        scheduler = CosineAnnealingLR(self._optimizer, T_max=self._epochs, eta_min=0.1 * self._lr)
        self.loss = []


        if not overwrite_checkpoint:
            save_iters = range(start_epoch, self._epochs, checkpoint_epoch)
            for iter_num in save_iters:
                ckpt_iter_name = f'{checkpoint_name}{iter_num}.pth'
                full_path = os.path.join(checkpoint_path, ckpt_iter_name)
                if os.path.exists(full_path):
                    raise FileExistsError(f"Checkpoint file already exists: {full_path}")

        progress_bar = tqdm(range(start_epoch, self._epochs), 
                            position=0,
                            desc="Training POTNet", 
                            disable=not self._verbose)
        for epoch in progress_bar:
            self._cur_epoch = epoch
            avg_loss = []
            for batch in dataloader:
                if self._conditional:
                    if use_weights:
                        conditioning_batch, data_batch, batch_weights = batch
                        batch_weights = batch_weights.to(self._device)
                    else:
                        conditioning_batch, data_batch = batch[:2]
                    conditioning_batch = conditioning_batch.to(self._device)
                else:
                    if use_weights:
                        data_batch, batch_weights = batch
                        batch_weights = batch_weights.to(self._device)
                    else:
                        data_batch = batch[0]
                    conditioning_batch = None

                data_batch = data_batch.to(self._device)
                batch_size = data_batch.shape[0]
                self._optimizer.zero_grad()
                noise_vec = torch.randn((batch_size, self._embedding_dim), device=self._device)

                if self._conditional:
                    generated_batch = self._generator(conditioning_batch, noise_vec)
                else:
                    generated_batch = self._generator(noise_vec)

                if use_weights:
                    ab = batch_weights / torch.sum(batch_weights)
                else:
                    ab = torch.ones(batch_size, device=self._device) / batch_size

                cost_matrix = torch.cdist(data_batch, generated_batch, p=self._p)
                joint_loss = ot.emd2(ab, ab, cost_matrix)

                loss_marginal = torch.sum(
                    self._mp_lambdas
                    * ot.wasserstein_1d(data_batch, generated_batch, ab, ab, self._p) ** (1 / self._p)
                )
                total_loss = joint_loss + loss_marginal
                total_loss.backward()
                self._optimizer.step()
                avg_loss.append(total_loss.item())
                del cost_matrix

            self.loss.append(np.mean(avg_loss))
            self._mp_lambdas *= self._mp_decay
            scheduler.step()
            
            progress_bar.set_postfix(loss=np.mean(avg_loss)) 

            if self._print_loss and (epoch + 1) % self._print_loss_iter == 0:
                print(f"Epoch {epoch + 1}/{self._epochs}, Loss: {self.loss[-1]}")

            if save_checkpoint and (epoch + 1) % checkpoint_epoch == 0:
                if checkpoint_name is None:
                    checkpoint_name = 'potnet_ckpt'
                self.save_checkpoint(path=checkpoint_path, 
                                     name=checkpoint_name,
                                     iteration=epoch + 1)

        if self._verbose:
            print("Training complete.")


    def generate(self, num_samples, cond_data=None):
        """
        Generates samples from the trained model.
        num_samples: Number of samples to generate.
        cond_data: Conditioning data for the conditional variant of the model.
        """
        self._generator.eval()
        if self._conditional:
            if cond_data is None:
                cond_data = self._cond_data
                num_samples = self._cond_data.shape[0]
                warnings.warn(f"No conditioning data provided. Generating {num_samples} samples using stored conditional data.")
            else:
                if num_samples != cond_data.shape[0]:
                    num_samples = cond_data.shape[0]
                    warnings.warn(f"Mismatch in sample and conditional data count. Generating {num_samples} samples.")
                cond_data = torch.tensor(
                    self._cond_data_transformer.standardize(cond_data),
                    dtype=torch.float32,
                    device=self._device
                )
            noise = torch.randn((num_samples, self._embedding_dim), device=self._device)
            generated_data = self._generator(noise, cond_data)
        else:
            noise = torch.randn((num_samples, self._embedding_dim), device=self._device)
            generated_data = self._generator(noise)

        generated_data = generated_data.detach().cpu().numpy()

        if self._num_disc_cols > 0:
            softmax_data = generated_data[:, :self._num_disc_cols]
            num_data = generated_data[:, self._num_disc_cols:]
            if num_data.shape[1] == 0:
                num_data = None
            else:
                num_data = self._transformer.inv_transform(num_data)
                num_data = self._transformer.output_transform(num_data)
            reconstructed_data = self._disc_data_transformer.inverse_encode(softmax_data, num_data)
            return reconstructed_data
        else:
            generated_data = self._transformer.inv_transform(generated_data)
            generated_data = self._transformer.output_transform(generated_data)
        return generated_data
    

    def save(self, model_path):
        """
        Save the entire model, including transformation modules and constants.
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_path}")



    def save_checkpoint(self, 
                        path='./', 
                        name='potnet_ckpt_iter',
                        iteration=None):
        """
        Saves a checkpoint containing the model and optimizer states.

        Args:
            path (str): Directory path for saving the checkpoint.
            name (str): Name of the checkpoint file.
            iteration (int): Iteration number for the checkpoint.
        """
        if path[-1] != '/':
            path += '/'

        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint_name = f'{name}{iteration}.pth' if iteration is not None else 'potnet_ckpt_iter0.pth'
        checkpoint = {
            'output_dim': self._output_dim,
            'architecture': self._architecture,
            'conditional': self._conditional,
            'cond_dim': self._cond_dim if self._conditional else None,
            'generator_state_dict': self._generator.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epoch': self._cur_epoch,
            'loss': self.loss,
            'class_attributes': self.__dict__
        }

        torch.save(checkpoint, path + checkpoint_name)
        print(f"Checkpoint saved to {path + checkpoint_name}")


    def load_checkpoint(self, checkpoint_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Loads a checkpoint and restores the model, optimizer, and training metadata.
        Updates the current model instance directly.
        Args:
            checkpoint_path: File path of the checkpoint.
            device: Torch device for loading (CPU or GPU).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

             # Update generator state
            if 'output_dim' in checkpoint:
                self._output_dim = checkpoint['output_dim']
            else:
                raise KeyError("Checkpoint does not contain output_dim.")

            if 'architecture' in checkpoint:
                self._architecture = checkpoint['architecture']
            else:
                raise KeyError("Checkpoint does not contain architecture.")
            
            if 'conditional' in checkpoint:
                self._conditional = checkpoint['conditional']
            else:
                raise KeyError("Checkpoint does not contain conditional.")
            
            if self._conditional:
                if 'cond_dim' in checkpoint:
                    self._cond_dim = checkpoint['cond_dim']
                else:
                    raise KeyError("Checkpoint does not contain cond_dim.")
            
            if self._generator is None:
                # Initialize the generator
                if self._conditional:
                    self._generator = POTNetCondGenerator(
                        self._embedding_dim,
                        self._cond_dim,
                        self._output_dim,
                        self._num_disc_cols,
                        self._architecture,
                        self._disc_col_category_idx,
                        self._activation,
                        self._batch_norm,
                        self._dropout,
                        self._dropout_rate,
                        self._skip
                    ).to(device)
                else:
                    self._generator = POTNetGenerator(
                        self._embedding_dim,
                        self._output_dim,
                        self._num_disc_cols,
                        self._architecture,
                        self._disc_col_category_idx,
                        self._activation,
                        self._batch_norm,
                        self._dropout,
                        self._dropout_rate,
                        self._skip
                    ).to(device)
                self._optimizer = torch.optim.AdamW(self._generator.parameters(), lr=self._lr, betas=self._betas)

            # Update generator state
            if 'generator_state_dict' in checkpoint:
                self._generator.load_state_dict(checkpoint['generator_state_dict'])
                self._generator = self._generator.to(device)
            else:
                raise KeyError("Checkpoint does not contain generator_state_dict.")

            # Update optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                raise KeyError("Checkpoint does not contain optimizer_state_dict.")

            # Update all attributes from `class_attributes`
            if 'class_attributes' in checkpoint:
                self.__dict__.update(checkpoint['class_attributes'])
            else:
                print("Warning: No class attributes found in checkpoint.")

            # Retrieve loss and epoch information
            self.loss = checkpoint.get('loss', [])
            self._cur_epoch = checkpoint.get('epoch', 0)

            print(f"Checkpoint successfully loaded from {checkpoint_path}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise





def load_model(model_path):
    """
    Load a model, including transformation modules and constants.
    Args:
        model_path: Path to the saved model file.
    Returns:
        The loaded model instance.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model
