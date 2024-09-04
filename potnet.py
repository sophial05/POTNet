"""POTNet module."""


import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import warnings

from .data_transformer import DataTransformer, DiscreteDataTransformer

class POTNetGenerator(nn.Module):
    """POTNetGenerator module for POTNet."""

    def __init__(self, 
                 embedding_dim,
                 output_dim,
                 num_disc_cols,
                 architecture,
                 disc_col_category_idx = None,
                 activation=nn.ReLU(inplace=True),
                 batch_norm=True,
                 dropout=True,
                 dropout_rate=0.1):
        super(POTNetGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.activation = activation
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.num_disc_cols = num_disc_cols
        self.disc_col_category_idx = disc_col_category_idx
        
        if architecture[0] != self.embedding_dim:
            raise ValueError("First layer of architecture must match input dimension.")
        
        if architecture[-1] != self.output_dim:
            raise ValueError("Last layer of architecture must match output dimension.")

        num_layers = len(self.architecture)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(self.architecture[i], self.architecture[i+1]))
            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(self.architecture[i+1]))
            self.layers.append(self.activation)
            if self.dropout:
                self.layers.append(nn.Dropout(p = self.dropout_rate))
        self.layers.append(nn.Linear(self.architecture[num_layers - 2], architecture[num_layers - 1]))

    # discrete data should always come first before numeric data
    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        for layer in self.layers:
            x = layer(x)
        if self.num_disc_cols > 0:
            prev_c = 0
            for c in range(len(self.disc_col_category_idx)):
                cur_idx = self.disc_col_category_idx[c]
                prev_idx = self.disc_col_category_idx[prev_c]
                x[:, prev_idx:cur_idx] = F.softmax(x[:, prev_idx:cur_idx], 
                                                          dim=1)
                prev_c = c
        return x
    
class POTNetCondGenerator(nn.Module):
    """POTNetCondGenerator module for POTNet."""

    def __init__(self, 
                 embedding_dim,
                 cond_dim,
                 output_dim,
                 num_disc_cols,
                 architecture,
                 disc_col_category_idx = None,
                 activation=nn.ReLU(inplace=True),
                 batch_norm=True,
                 dropout=True,
                 dropout_rate=0.1):
        super(POTNetCondGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        self.input_dim = self.embedding_dim + self.cond_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.activation = activation
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.num_disc_cols = num_disc_cols
        self.disc_col_category_idx = disc_col_category_idx
        
        if architecture[0] != self.input_dim:
            raise ValueError("First layer of architecture must match input dimension.")
        
        if architecture[-1] != self.output_dim:
            raise ValueError("Last layer of architecture must match output dimension.")

        num_layers = len(self.architecture)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(self.architecture[i], self.architecture[i+1]))
            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(self.architecture[i+1]))
            self.layers.append(self.activation)
            if self.dropout:
                self.layers.append(nn.Dropout(p = self.dropout_rate))
        self.layers.append(nn.Linear(self.architecture[num_layers - 2], architecture[num_layers - 1]))

    def forward(self, x, z):
        input = torch.cat((x, z), dim=1)
        input = input.view(-1, self.input_dim)
        for layer in self.layers:
            input = layer(input)
        
        # discrete data should always come first before numeric data
        if self.num_disc_cols > 0:
            prev_c = 0
            for c in range(len(self.disc_col_category_idx)):
                cur_idx = self.disc_col_category_idx[c]
                prev_idx = self.disc_col_category_idx[prev_c]
                input[:, prev_idx:cur_idx] = F.softmax(input[:, prev_idx:cur_idx], 
                                                              dim=1)
                prev_c = c
        return input
    

class POTNetDataset(Dataset):
    """Dataset class for POTNet."""

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class POTNetConditionalDataset(Dataset):
    """Dataset class for POTNet."""

    def __init__(self, conditioning_data, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.conditioning_data = torch.tensor(conditioning_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.conditioning_data[idx], self.data[idx], 


class POTNet():
    """Penalized Optimal Transport Network (POTNet) class."""

    def __init__(self,
                 embedding_dim = None,
                 architecture = (256, 256),
                 conditional=False,
                 categorical_cols = None, # discrete columns in data (excluding conditional data)
                 numeric_output_data_type = 'continuous', # 'integer' or 'continuous', 
                            # or dict of type and associated columns {'integer': [col1, col2], 'continuous': [col3, col4]}
                 standardize=True,
                 p = 1.0,
                 mp_lambdas = None,
                 activation=nn.ReLU(inplace=True),
                 batch_norm=True,
                 dropout=True,
                 dropout_rate=0.1,
                 epochs=300,
                 batch_size=256,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 verbose=False):
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
        self._activation = activation
        self._dropout = dropout
        self._dropout_rate = dropout_rate
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
        
        if isinstance(numeric_output_data_type, dict):
            if not all(key in ['integer', 'continuous'] for key in numeric_output_data_type):
                raise ValueError("Invalid output data type format: keys must be `integer` or `continuous`.")
        else:
            if numeric_output_data_type not in ['integer', 'continuous']:
                raise ValueError("numeric_output_data_type must be one of 'integer', 'discrete' or 'continuous'.")

        self.numeric_output_data_type = numeric_output_data_type

        if self._num_disc_cols > 0:
            self._transformer = DataTransformer(data_type = self.numeric_output_data_type, 
                                                transformation='min_max')
        else:
            if self._standardize:
                self._transformer = DataTransformer(data_type = self.numeric_output_data_type, 
                                                    transformation='standardize')
            else:
                self._transformer = DataTransformer(data_type = self.numeric_output_data_type, 
                                                    transformation=None)
            

        self._data_sampler = None
        self._generator = None
        self._optimizer = None


    @staticmethod
    def _init_weights_normal(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 1/np.sqrt(10000))

    @staticmethod
    def _check_interactive():
        '''Check if running in interactive mode.'''
        import __main__ as main
        return not hasattr(main, '__file__')
    

    def get_loader(self, data, conditioning_data=None):
        """Get data loader for POTNet.

        Args:
            data (numpy.ndarray or pandas.DataFrame): Data to be loaded.
        
        Returns:
            torch.utils.data.DataLoader
        """
        if isinstance(self._mp_lambdas, float):
            self._mp_lambdas = np.ones(data.shape[1]) * self._mp_lambdas

        if self._mp_lambdas is not None and self._mp_lambdas is not float and len(self._mp_lambdas) != data.shape[1]:
            raise ValueError("mp_lambdas must be of same length as the data dimension.") 
        

        if self._categorical_cols is not None:
            disc_data, num_data, self._mp_lambdas, \
                num_disc_cols, \
                disc_col_category_idx = self._disc_data_transformer.one_hot_encode(data, 
                                                                                   self._categorical_cols,
                                                                                   self._mp_lambdas)
            self._disc_col_category_idx = disc_col_category_idx
            self._num_disc_cols = num_disc_cols

        if self._num_disc_cols > 0:
            if num_data is not None:
                num_data = self._transformer.standardize(num_data, 
                                                         column_names=self._disc_data_transformer._cont_data_columns)
        else:
            if isinstance(data, pd.DataFrame):
                data = data.astype(float).to_numpy()
            data = self._transformer.standardize(data)

        if self._conditional:
            if self._num_disc_cols > 0:
                self._cond_data_transformer = DataTransformer(data_type = 'continuous', transformation='min_max')
                colnames = self._disc_data_transformer._cont_data_columns
            else:
                if self._standardize:
                    self._cond_data_transformer = DataTransformer(data_type = 'continuous', transformation='standardize')
                else:
                    self._cond_data_transformer = DataTransformer(data_type = 'continuous', transformation=None)
                colnames = None
                conditioning_data = self._cond_data_transformer.standardize(conditioning_data,
                                                                            column_names=colnames)
            
            self._cond_data = conditioning_data
        

        if self._num_disc_cols > 0:
            # discrete data comes before numeric data
            if num_data is not None:
                data = np.concatenate((disc_data, num_data), axis=1)
            else:
                data = disc_data    


        if self._conditional:
            dataset = POTNetConditionalDataset(conditioning_data, 
                                               data)
        else:
            dataset = POTNetDataset(data)
        self._output_dim = data.shape[1]
        if self._embedding_dim is None:
            self._embedding_dim = self._output_dim
            
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

    
    def _contains_nan(self, data):
        if isinstance(data, np.ndarray):
            return np.isnan(data).any()
        elif isinstance(data, pd.DataFrame):
            return data.isna().any().any()
        else:
            raise TypeError("Input must be a NumPy ndarray or a Pandas DataFrame")


    def fit(self, train_data, 
            conditioning_data = None):
        """Fit POTNet to the training data.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Training data."""
        
        if self._contains_nan(train_data):
            raise ValueError("Training data contains missing values.")

        if self._verbose:
            print("Fitting POTNet...")
        if self._check_interactive():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm


        if self._conditional:
            self._cond_dim = conditioning_data.shape[1]
            dataloader = self.get_loader(train_data, conditioning_data)
        else:
            dataloader = self.get_loader(train_data)


        if self._conditional:
            self._architecture = [self._embedding_dim + self._cond_dim] + list(self._input_architecture) + [self._output_dim]
        else:
            self._architecture = [self._embedding_dim] + list(self._input_architecture) + [self._output_dim]

        
        if len(self._mp_lambdas) != self._output_dim:
            raise ValueError("mp_lambdas must be of length output_dim.")
        self._mp_lambdas = torch.as_tensor(self._mp_lambdas, device=self._device, dtype=torch.float32)


        if self._conditional:
            self._cond_data = Variable(torch.tensor(self._cond_data, dtype=torch.float32)).to(self._device)
            self._generator = POTNetCondGenerator(self._embedding_dim,
                                    self._cond_dim,
                                    self._output_dim,
                                    self._num_disc_cols,
                                    self._architecture,
                                    self._disc_col_category_idx,
                                    self._activation,
                                    self._batch_norm,
                                    self._dropout,
                                    self._dropout_rate).to(self._device)
        else:
            self._generator = POTNetGenerator(self._embedding_dim,
                                    self._output_dim,
                                    self._num_disc_cols,
                                    self._architecture,
                                    self._disc_col_category_idx,
                                    self._activation,
                                    self._batch_norm,
                                    self._dropout,
                                    self._dropout_rate).to(self._device)

        
        self._optimizer = torch.optim.AdamW(self._generator.parameters(),
                                          lr=self._lr,
                                          betas=self._betas)
        self._generator.apply(self._init_weights_normal)
        self._generator.train()

        if not self._conditional:
            self.loss = []
            for epoch in tqdm(range(self._epochs), desc="Training POTNet"):
                avg_loss = []
                for i, data_batch in enumerate(dataloader):
                    data_batch = Variable(data_batch).to(self._device)
                    batch_size = data_batch.shape[0]

                    self._optimizer.zero_grad()
                    noise_vec = torch.randn((batch_size, self._embedding_dim),
                                            device = self._device)
                    noise_vec = Variable(noise_vec)
                    generated_batch = self._generator(noise_vec)

                    ab = torch.ones(batch_size) / batch_size
                    ab = ab.to(self._device)

                    dist_mat = torch.cdist(data_batch, generated_batch, p=2.0)
                    joint_loss = ot.emd2(ab, ab, dist_mat)
                    avg_loss.append(float(joint_loss.detach()))
                    joint_loss.backward(retain_graph=True)
                    del dist_mat

                    loss_marginal = torch.sum(self._mp_lambdas * ot.wasserstein_1d(data_batch, generated_batch, ab, ab, self._p))
                    loss_marginal.backward(retain_graph=True)
                    avg_loss.append(float(loss_marginal.detach()))

                    self._optimizer.step()
                self.loss.append(np.mean(avg_loss))
                
                if self._verbose:
                    print("Epoch: {} | Loss: {}".format(epoch+1, self.loss[-1]))
            
            if self._verbose:
                print("Done fitting POTNet.")
        else:
            self.loss = []
            for epoch in tqdm(range(self._epochs), desc="Training POTNet"):
                avg_loss = []
                for i, (conditioning_batch, data_batch) in enumerate(dataloader):
                    conditioning_batch = Variable(conditioning_batch).to(self._device)
                    data_batch = Variable(data_batch).to(self._device)
                    batch_size = data_batch.shape[0]

                    self._optimizer.zero_grad()
                    noise_vec = torch.randn((batch_size, self._embedding_dim),
                                            device = self._device)
                    noise_vec = Variable(noise_vec)
                    generated_batch = self._generator(conditioning_batch, noise_vec)

                    ab = torch.ones(batch_size) / batch_size
                    ab = ab.to(self._device)

                    dist_mat = torch.cdist(data_batch, generated_batch, p=2.0)
                    joint_loss = ot.emd2(ab, ab, dist_mat)
                    avg_loss.append(float(joint_loss.detach()))
                    joint_loss.backward(retain_graph=True)
                    del dist_mat

                    loss_marginal = torch.sum(self._mp_lambdas * ot.wasserstein_1d(data_batch, generated_batch, ab, ab, self._p))
                    loss_marginal.backward(retain_graph=True)
                    avg_loss.append(float(loss_marginal.detach()))

                    self._optimizer.step()
                self.loss.append(np.mean(avg_loss))
                
                if self._verbose:
                    print("Epoch: {} | Loss: {}".format(epoch+1, self.loss[-1]))
            
            if self._verbose:
                print("Done fitting POTNet.")
    
    def generate(self, num_samples, cond_data = None):
        """Generate data similar to the training data.
        
        Args:
            num_samples (int): Number of samples to generate."""
        
        self._generator.eval()
        if self._conditional:
            if cond_data is None:
                cond_data = self._cond_data
                num_samples = self._cond_data.shape[0]
                warnings.warn("No conditioning data provided. \n Generating {} using previously inputted conditional data.".format(num_samples))
            else:
                if num_samples != cond_data.shape[0]:
                    num_samples = cond_data.shape[0]
                    warnings.warn("Number of samples does not match number of conditioning data. \n Generating {} using inputted conditional data.".format(num_samples))
                cond_data = torch.tensor(self._cond_data_transformer.standardize(cond_data),
                                            dtype=torch.float32, 
                                            device=self._device)
            
            noise = torch.randn((num_samples, self._embedding_dim), device=self._device)
            generated_data = self._generator(noise, cond_data)
        else:
            noise = torch.randn((num_samples, self._embedding_dim), device=self._device)
            generated_data = self._generator(noise)

        
        generated_data = generated_data.detach().cpu().numpy()

        # inverse transform
        if self._num_disc_cols > 0:
            softmax_data = generated_data[:, :self._num_disc_cols]
            num_data = generated_data[:, self._num_disc_cols:]
            if num_data.shape[1] ==  0:
                num_data = None
            else:
                num_data = self._transformer.inv_transform(num_data)
                # transform to correct data type
                num_data = self._transformer.output_transform(num_data)
            reconstructed_data = self._disc_data_transformer.inverse_encode(softmax_data, num_data)
            return reconstructed_data
        else:
            generated_data = self._transformer.inv_transform(generated_data)
            # transform to correct data type
            generated_data = self._transformer.output_transform(generated_data)
            
        return generated_data
        
    def save(self, 
             model_name=None,
             path='./'):
        """Save trained model.
        
        Args:
            path (str): Path to save model."""
        
        if path[-1] != '/':
            path += '/'
        if model_name is None and path.find('pt') == -1:
            model_name = 'potnet.pt'
        torch.save(self._generator, path + model_name)
        np.savetxt(path + 'potnet_loss.txt', self.loss)

