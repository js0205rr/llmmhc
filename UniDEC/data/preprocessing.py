"""
Data utilities for extreme classification.

This module provides utilities for data loading, preprocessing, and tokenization
for extreme classification tasks. It supports both short and long text data formats
and includes functionality for inverse propensity scoring.
"""

import warnings
import os
import json
import re
import numpy as np
import torch
import pickle as pkl
from scipy import sparse as sp
from xclib.data import data_utils as du
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Union, Optional


# Pre-trained model configurations
TOKENIZERS = {
    'distilbert': "sentence-transformers/msmarco-distilbert-base-v4",
    'phi3': "microsoft/Phi-3-mini-128k-instruct"
}

# Dataset-specific parameters for inverse propensity scoring
DATASET_PARAMS = {
    'AmazonTitles-670K': {'A': 0.6, 'B': 2.6},
    'AmazonTitles-3M': {'A': 0.6, 'B': 2.6},
    'WikiSeeAlsoTitles-350K': {'A': 0.55, 'B': 1.5},
    'LF-WikiSeeAlso-320K': {'A': 0.55, 'B': 1.5},
    'WikiTitles-500K': {'A': 0.5, 'B': 0.4},
    'LF-WikiTitles-500K': {'A': 0.55, 'B': 0.55},
    'LF-AmazonTitles-1.3M': {'A': 0.6, 'B': 2.6}
}


def encode(sent: Union[str, List[str]], params) -> Dict[str, List[int]]:
    """
    Encode text using the specified tokenizer.
    
    Args:
        sent: Input text or list of tokens
        params: Parameters containing sequence length and other settings
        
    Returns:
        Dictionary containing input_ids and attention_mask
    """
    if isinstance(sent, list):
        sent = ' '.join(sent)
    return tokenizer(
        sent, 
        truncation=True, 
        padding='max_length', 
        max_length=params.seq_len
    )


def create_data(args) -> Tuple[Dict[str, torch.Tensor], sp.spmatrix, 
                             Dict[str, torch.Tensor], sp.spmatrix, 
                             Dict[str, torch.Tensor], np.ndarray]:
    """
    Create and tokenize dataset.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        Tuple of (X_trn, Y_trn, X_tst, Y_tst, X_lbl, inv_prop)
    """
    # Load raw data
    if 'Titles' in args.dataset:
        trn_sents, tst_sents, lbl_feats, inv_prop = load_short_data_raw(args)
    else:
        trn_sents, tst_sents, lbl_feats, inv_prop = load_long_data_raw(args)
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZERS[args.pre_trained_model],
        cache_dir='/scratch/project_2001083/devaansh/hunggingface_extra_cache'
    )
    
    # Initialize data containers
    X_trn = {'input_ids': [], 'attention_mask': []}
    X_tst = {'input_ids': [], 'attention_mask': []}
    X_lbl = {'input_ids': [], 'attention_mask': []}
    
    # Tokenize training data
    for sent in tqdm(trn_sents, desc='Tokenizing train data'):
        encoded = encode(sent, args)
        X_trn['input_ids'].append(encoded['input_ids'])
        X_trn['attention_mask'].append(encoded['attention_mask'])
    
    # Tokenize test data
    for sent in tqdm(tst_sents, desc='Tokenizing test data'):
        encoded = encode(sent, args)
        X_tst['input_ids'].append(encoded['input_ids'])
        X_tst['attention_mask'].append(encoded['attention_mask'])
    
    # Tokenize label data
    for sent in tqdm(lbl_feats, desc='Tokenizing label data'):
        encoded = encode(sent, args)
        X_lbl['input_ids'].append(encoded['input_ids'])
        X_lbl['attention_mask'].append(encoded['attention_mask'])
    
    # Create output directory
    os.makedirs(os.path.join(args.data_path, args.pre_trained_model), exist_ok=True)
    
    # Convert to numpy arrays and save
    for key in X_trn.keys():
        X_trn[key] = np.stack(X_trn[key])
        X_tst[key] = np.stack(X_tst[key])
        X_lbl[key] = np.stack(X_lbl[key])
        
        # Save numpy arrays
        np.save(os.path.join(args.data_path, args.pre_trained_model, f'x_train_{key}.npy'), X_trn[key])
        np.save(os.path.join(args.data_path, args.pre_trained_model, f'x_test_{key}.npy'), X_tst[key])
        np.save(os.path.join(args.data_path, args.pre_trained_model, f'x_lbl_{key}.npy'), X_lbl[key])
        
        # Convert to torch tensors
        X_trn[key] = torch.from_numpy(X_trn[key])
        X_tst[key] = torch.from_numpy(X_tst[key])
        X_lbl[key] = torch.from_numpy(X_lbl[key])
    
    # Load label matrices
    Y_trn = du.read_sparse_file(os.path.join(args.data_path, 'trn_X_Y.txt'))
    Y_tst = du.read_sparse_file(os.path.join(args.data_path, 'tst_X_Y.txt'))
    
    return X_trn, Y_trn, X_tst, Y_tst, X_lbl, inv_prop


def load_data(args) -> Tuple[Dict[str, torch.Tensor], sp.spmatrix, 
                           Dict[str, torch.Tensor], sp.spmatrix, 
                           Dict[str, torch.Tensor], np.ndarray]:
    """
    Load preprocessed dataset.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        Tuple of (X_trn, Y_trn, X_tst, Y_tst, X_lbl, inv_prop)
    """
    # Load label matrices
    if os.path.exists(os.path.join(args.data_path, 'trn_X_Y.npz')):
        Y_trn = sp.load_npz(os.path.join(args.data_path, 'trn_X_Y.npz'))
        Y_tst = sp.load_npz(os.path.join(args.data_path, 'tst_X_Y.npz'))
    else:
        Y_trn = du.read_sparse_file(os.path.join(args.data_path, 'trn_X_Y.txt'))
        Y_tst = du.read_sparse_file(os.path.join(args.data_path, 'tst_X_Y.txt'))
        sp.save_npz(os.path.join(args.data_path, 'trn_X_Y.npz'), Y_trn)
        sp.save_npz(os.path.join(args.data_path, 'tst_X_Y.npz'), Y_tst)
    
    # Initialize data containers
    X_trn = {'input_ids': None, 'attention_mask': None}
    X_tst = {'input_ids': None, 'attention_mask': None}
    X_lbl = {'input_ids': None, 'attention_mask': None}
    
    # Load tokenized data
    model_path = os.path.join(args.data_path, args.pre_trained_model)
    for key in ['input_ids', 'attention_mask']:
        X_trn[key] = torch.from_numpy(np.load(os.path.join(model_path, f'trn_doc_{key}.npy')))
        X_tst[key] = torch.from_numpy(np.load(os.path.join(model_path, f'tst_doc_{key}.npy')))
        X_lbl[key] = torch.from_numpy(np.load(os.path.join(model_path, f'lbl_{key}.npy')))
    
    # Load or compute inverse propensity scores
    inv_prop_path = os.path.join(args.data_path, 'inv_prop.npy')
    if os.path.exists(inv_prop_path):
        inv_prop = np.load(inv_prop_path)
    else:
        inv_prop = get_inv_prop(Y_trn, args)
        np.save(inv_prop_path, inv_prop)
    
    return X_trn, Y_trn, X_tst, Y_tst, X_lbl, inv_prop


def load_short_data_raw(args) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Load short text data from raw files.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        Tuple of (trn_sents, tst_sents, lbl_data, inv_prop)
    """
    raw_data_path = os.path.join(args.data_path, 'raw')
    trn_data, trn_labels = [], []
    tst_data, tst_labels = [], []
    lbl_data = []
    
    # Load training data
    with open(os.path.join(raw_data_path, 'trn.json')) as fin:
        for info in tqdm(fin.readlines(), desc='Reading training data'):
            info = json.loads(info)
            trn_data.append(info['title'])
            trn_labels.append(np.array(info['target_ind']))
    
    # Load test data
    with open(os.path.join(raw_data_path, 'tst.json'), 'r') as fin:
        for info in tqdm(fin.readlines(), desc='Reading testing data'):
            info = json.loads(info)
            tst_data.append(info['title'])
    
    # Load label data
    with open(os.path.join(raw_data_path, 'lbl.json'), 'r') as fin:
        for info in tqdm(fin.readlines(), desc='Reading labels data'):
            info = json.loads(info)
            lbl_data.append(info['title'])
    
    assert len(trn_data) == len(trn_labels)
    
    # Clean and process data
    trn_sents = data_cleaner(trn_data)
    tst_sents = data_cleaner(tst_data)
    lbl_data = data_cleaner(lbl_data)
    
    inv_prop = get_inv_prop(trn_labels, args)
    return trn_sents, tst_sents, lbl_data, inv_prop


def load_long_data_raw(args) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Load long text data from raw files.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        Tuple of (trn_sents, tst_sents, lbl_data, inv_prop)
    """
    trn_data, trn_labels = [], []
    tst_data, tst_labels = [], []
    lbl_data = []
    
    # Load training data
    with open(os.path.join(args.data_path, 'trn.json')) as fin:
        for info in tqdm(fin.readlines(), desc='Reading training data'):
            info = json.loads(info)
            trn_data.append(info['content'])
            trn_labels.append(np.array(info['target_ind']))
    
    # Load test data
    with open(os.path.join(args.data_path, 'tst.json'), 'r') as fin:
        for info in tqdm(fin.readlines(), desc='Reading testing data'):
            info = json.loads(info)
            tst_data.append(info['content'])
    
    # Load label data
    with open(os.path.join(args.data_path, 'lbl.json'), 'r') as fin:
        for info in tqdm(fin.readlines(), desc='Reading labels data'):
            info = json.loads(info)
            lbl_data.append(info['title'])
    
    assert len(trn_data) == len(trn_labels)
    
    # Clean and process data
    trn_sents = data_cleaner(trn_data)
    tst_sents = data_cleaner(tst_data)
    lbl_data = data_cleaner(lbl_data)
    
    inv_prop = get_inv_prop(trn_labels, args)
    return trn_sents, tst_sents, lbl_data, inv_prop


def get_inv_prop(Y: sp.spmatrix, args) -> np.ndarray:
    """
    Compute inverse propensity scores.
    
    Args:
        Y: Label matrix
        args: Arguments containing dataset configuration
        
    Returns:
        Array of inverse propensity scores
    """
    print("Creating inv_prop file")
    
    if not os.path.exists(os.path.join(args.data_path, 'inv_prop.npy')):
        params = DATASET_PARAMS[args.dataset]
        a, b = params['A'], params['B']
        
        num_labels = Y.shape[1]
        num_samples = Y.shape[0]
        inv_prop = np.array(Y.sum(axis=0)).ravel()
        
        c = (np.log(num_samples) - 1) * np.power(b+1, a)
        inv_prop = 1 + c * np.power(inv_prop + b, -a)
    else:
        inv_prop = np.load(os.path.join(args.data_path, 'inv_prop.npy'))
    
    return inv_prop


def data_cleaner(data: List[str]) -> List[List[str]]:
    """
    Clean and tokenize text data.
    
    Args:
        data: List of text strings
        
    Returns:
        List of tokenized and cleaned text
    """
    for i, t in enumerate(data):
        data[i] = clean_str(t)
        if len(data[i]) == 0:
            data[i] = t.split()
    
    return data


def clean_str(string: str) -> List[str]:
    """
    Clean and tokenize a single string.
    
    Args:
        string: Input text string
        
    Returns:
        List of cleaned tokens
    """
    # Replace underscores with spaces
    string = re.sub(r"_", " ", string)
    
    # Handle punctuation
    string = re.sub('(?<=[A-Za-z]),', ' ', string)
    string = re.sub(r"(),!?", "", string)
    string = re.sub(r"[^A-Za-z0-9\.\'\`]", " ", string)
    string = re.sub('(?<=[A-Za-z])\.', '', string)
    
    # Handle possessives
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"s\' ", " ", string)
    
    # Clean up whitespace
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    
    return string.split()
