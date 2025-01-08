import numpy as np 
import pandas as pd
from functools import reduce

class DataTransformer(object):
    """Data transformer for POTNet."""
    
    def __init__(self, 
                 data_type = 'continuous',
                 transformation = 'standardize'): # 'min_max' or 'standardize' or None
        self._shift = None
        self._scale = None
        self._min = None
        self._max = None
        self._transformation = transformation
        self.data_type = data_type
        self._column_names = None
        self._transform_fn = None
        if self.data_type == 'integer':
            self._transform_fn = [np.round, np.int32]
        elif self.data_type == 'continuous': # default: 'continuous'
            self._transform_fn = [np.float32]
        else:
            self._transform_fn = {}
            for dtype, cols in self.data_type.items():
                if dtype == 'integer':
                    for col in cols:
                        self._transform_fn[col] = [np.round, np.int32]
                elif dtype == 'continuous':
                    for col in cols:
                        self._transform_fn[col] = [np.float32]
        

    def _fit(self, data):
        """Fit the transformer to the data."""
        if self._transformation == 'standardize':
            self._shift = data.mean(axis=0)
            self._scale = data.std(axis=0)
        elif self._transformation == 'min_max':
            self._min = data.min(axis=0)
            self._max = data.max(axis=0)
        else:
            pass # do not do anything

    def standardize(self, data, column_names=None):
        """Standardize the data."""
        if type(data) == pd.DataFrame:
            self._column_names = data.columns
            data = data.to_numpy()
        if column_names is not None:
            self._column_names = column_names
        self._fit(data)
        if self._transformation == 'standardize':
            return (data - self._shift) / self._scale
        elif self._transformation == 'min_max':
            return (data - self._min) / (self._max - self._min)
        else:
            return data
    
    def inv_transform(self, data):
        """Inverse standardization of the data."""
        if self._transformation == 'standardize':
            if self._scale is not None and self._shift is not None:
                data = data * self._scale + self._shift
        elif self._transformation ==  'min_max':
            if self._min is not None and self._max is not None:
                data = data * (self._max - self._min) + self._min
        else:
            pass # do not do anything
                
        if self._column_names is not None:
            data = pd.DataFrame(data, columns=self._column_names)
        return data
    

    def output_transform(self, data):
        """Transform the data."""
        if isinstance(self._transform_fn, dict):
            for col, fn in self._transform_fn.items():
                if isinstance(col, str) and isinstance(data, np.ndarray):
                    if self._column_names is None:
                        raise ValueError('Continuous data column names must be provided.')
                    col = self._column_names.get_loc(col)
                    data[:, col] = reduce(lambda x, fn: fn(x), fn, data[:, col])
                elif isinstance(col, int) and isinstance(data, pd.DataFrame):
                    data.iloc[:, col] = reduce(lambda x, fn: fn(x), fn, data.iloc[:, col])
                elif isinstance(col, str) and isinstance(data, pd.DataFrame):
                    data[col] = reduce(lambda x, fn: fn(x), fn, data[col])
                elif isinstance(col, int) and isinstance(data, np.ndarray):
                    data[:, col] = reduce(lambda x, fn: fn(x), fn, data[:, col])
                else:
                    raise ValueError('Invalid input data type.')
            return data
        else:
            data = reduce(lambda x, fn: fn(x), self._transform_fn, data)
            return data
        


class DiscreteDataTransformer(object):
    """Data transformer for discrete data."""

    def __init__(self, 
                 categorical_cols):
        self._categorical_cols = categorical_cols
        self._mapping = None
        self._disc_col_category_indx = None
        self._onehot_columns = None
        self._onehot_index = None
        self._cont_data_columns = None
        self._orig_data_columns = None
    
    def _check_colnames(self, categorical_cols):
        if np.all([isinstance(col_name, str) for col_name in categorical_cols]):
            return 'str'
        if np.all([isinstance(col_name, int) for col_name in categorical_cols]):
            return 'int'
        else:
            raise ValueError('Column names must be either all strings or all integers.')


    def _get_variable_importance(self, data, cont_data):
        """Get variable importance scores."""
        if cont_data is None:
            return np.ones(data.shape[1])
        
        mp_lambdas = np.std(cont_data, axis=0)
        mp_lambdas = mp_lambdas / np.sum(mp_lambdas) * len(mp_lambdas)
        mp_lambdas[mp_lambdas == 0] = min(mp_lambdas[mp_lambdas > 0])
        mp_input = np.ones(data.shape[1])
        columns = list(data.columns)
        cont_col_idx = [columns.index(x) for x in columns if x not in self._categorical_cols]
        mp_input[cont_col_idx] = mp_lambdas
        return mp_input
    

    def one_hot_encode(self, data, categorical_cols, mp_lambdas):
        data = pd.DataFrame(data)
        data.columns = data.columns.astype('str')
        if pd.isnull(data).any().any():
            raise ValueError('Data contains missing values.')
        self._orig_data_columns = data.columns  
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].str.replace('_', '#')
        if self._check_colnames(categorical_cols) == 'int':
            categorical_cols = data.columns[categorical_cols]
        continuous_data = data.drop(categorical_cols, axis=1).astype('float32')
        if len(continuous_data.columns) == 0:
            continuous_data = None
        else:
            self._cont_data_columns = continuous_data.columns
            continuous_data = continuous_data.to_numpy()

        compute_mp_lambda = False
        if mp_lambdas is None:
            compute_mp_lambda = True
            mp_lambdas = self._get_variable_importance(data, continuous_data)
        
        one_hot_enc_data = pd.get_dummies(data[categorical_cols].astype('category'))
        self._onehot_columns = one_hot_enc_data.columns
        self._onehot_index = one_hot_enc_data.index
        
        # mapping for each one-hot encoded column to original column
        mapping = {col: orig_col for orig_col in categorical_cols for col in self._onehot_columns 
                   if col.startswith(orig_col)}
        self._mapping = mapping
        
        encoded_mp_lambda = []
        if continuous_data is not None:
            encoded_mp_lambda.extend(mp_lambdas[:continuous_data.shape[1]])

        if compute_mp_lambda:
            one_hot_mp_lambdas = np.std(one_hot_enc_data, axis=0)
            one_hot_mp_lambdas = one_hot_mp_lambdas / np.sum(one_hot_mp_lambdas) * len(one_hot_mp_lambdas)
            one_hot_mp_lambdas[one_hot_mp_lambdas == 0] = min(one_hot_mp_lambdas[one_hot_mp_lambdas > 0])
            encoded_mp_lambda.extend(one_hot_mp_lambdas)

            encoded_mp_lambda = np.array(encoded_mp_lambda) / np.sum(encoded_mp_lambda) * len(encoded_mp_lambda)
        else:
            for col in one_hot_enc_data.columns:
                encoded_mp_lambda.append(
                    mp_lambdas[self._orig_data_columns.tolist().index(mapping[col])])
        
            
        num_disc_cols = len(mapping)
        col_indx = self.get_category_end_indices()
        self._disc_col_category_indx = col_indx
        
        one_hot_enc_data = one_hot_enc_data.to_numpy()
        encoded_mp_lambda = np.array(encoded_mp_lambda)
        return one_hot_enc_data, continuous_data, encoded_mp_lambda, num_disc_cols, col_indx
    


    


    def softmax_to_one_hot(self, softmax_data):
        one_hot_matrix = np.zeros_like(softmax_data)
        start_idx = 0
        for end_idx in self._disc_col_category_indx:
            argmax_indices = np.argmax(softmax_data[:, start_idx:end_idx], axis=1)
            for row, col in enumerate(argmax_indices):
                one_hot_matrix[row, start_idx + col] = 1
            start_idx = end_idx
        return one_hot_matrix

        
    def inverse_encode(self, softmax_data, continuous_data):
        one_hot_enc_data = self.softmax_to_one_hot(softmax_data)
        one_hot_enc_data = pd.DataFrame(one_hot_enc_data, 
                                        columns=self._onehot_columns, 
                                        index=np.arange(len(one_hot_enc_data)))
        reverse_mapping = {}
        for k, v in self._mapping.items():
            reverse_mapping.setdefault(v, []).append(k)
        
        reconstructed_data = pd.DataFrame(index=one_hot_enc_data.index)
        for orig_col, cols in reverse_mapping.items():
            reconstructed_data[orig_col] = one_hot_enc_data[cols].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
        for col in reconstructed_data.columns:
            if reconstructed_data[col].dtype == 'object':
                reconstructed_data[col] = reconstructed_data[col].str.replace('#', '_')
        if continuous_data is not None:
            continuous_data = pd.DataFrame(continuous_data, columns=self._cont_data_columns).astype('float32')
            reconstructed_data = pd.concat([reconstructed_data, continuous_data], axis=1)
        reconstructed_data = reconstructed_data[self._orig_data_columns]
        return reconstructed_data
        

    def get_category_end_indices(self):
        reverse_mapping = {}
        for col, cat in self._mapping.items():
            reverse_mapping.setdefault(cat, []).append(col)

        end_indices = {}
        for cat, cols in reverse_mapping.items():
            end_indices[cat] = self._onehot_columns.get_loc(cols[-1]) + 1

        sorted_end_indices = [end_indices[cat] for cat in sorted(end_indices, key=lambda x: end_indices[x])]
        return sorted_end_indices

