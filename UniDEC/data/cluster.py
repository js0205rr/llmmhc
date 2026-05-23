"""
Clustering module for hierarchical label clustering.

This module implements hierarchical clustering algorithms for large-scale multi-label classification,
particularly for extreme classification problems. It provides functionality to build label trees
by clustering labels based on their feature representations.
"""

import os
import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer


def get_sparse_feature(feature_file: str, label_file: str) -> tuple:
    """
    Load and process sparse features and labels from files.

    Args:
        feature_file (str): Path to the feature file in SVMLight format
        label_file (str): Path to the label file containing space-separated labels

    Returns:
        tuple: Normalized sparse features and array of labels
    """
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    sparse_labels = [i.replace('\n', '').split() for i in open(label_file)]
    return normalize(sparse_x), np.array(sparse_labels)


def build_tree_by_level(sparse_data_x: str, sparse_data_y: str, eps: float, max_leaf: int, 
                       levels: list, groups_path: str) -> MultiLabelBinarizer:
    """
    Build a hierarchical tree structure by clustering labels.

    Args:
        sparse_data_x (str): Path to feature file
        sparse_data_y (str): Path to label file
        eps (float): Convergence threshold for clustering
        max_leaf (int): Maximum number of labels in a leaf node
        levels (list): List of levels to save intermediate results
        groups_path (str): Path to save the clustering results

    Returns:
        MultiLabelBinarizer: Fitted label binarizer
    """
    print('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_y = mlb.fit_transform(sparse_labels)
    joblib.dump(mlb, groups_path+'mlb')
    
    print('Getting Labels Feature')
    labels_f = normalize(sparse_y.T @ csc_matrix(sparse_x))
    print(f'Start Clustering {levels}')
    
    # Initialize levels and queue
    levels, q = [2**x for x in levels], None
    
    # Try to load existing clustering results
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(f'{groups_path}-Level-{i}.npy'):
            print(f'{groups_path}-Level-{i}.npy')
            labels_list = np.load(f'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    
    # Main clustering loop
    num_split = len([1 for node_i,_ in q if len(node_i) > max_leaf])
    while num_split:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        
        # Save intermediate results if current level matches target levels
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(f'Finish Clustering Level-{level}')
            np.save(f'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        
        # Process each node
        next_q = []
        max_size = max([len(node_i) for node_i, _ in q])
        print(f'Maximum size of node is {max_size}')
        
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
            else:
                next_q += [(node_i, node_f)]
        
        q = next_q
        print(f'Size of next_q {len(q)}')
        num_split = len([1 for node_i,_ in q if len(node_i) > max_leaf])
        print(f'Number of nodes to split is {num_split}')
        print()
    
    # Save final results
    labels_list = np.asarray([x[0] for x in q])
    np.save(f'{groups_path}-last.npy', np.asarray(labels_list))
    
    print('Finish Clustering')
    return mlb


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float) -> tuple:
    """
    Split a node into two clusters using k-means-like algorithm.

    Args:
        labels_i (np.ndarray): Indices of labels in the current node
        labels_f (csr_matrix): Feature matrix for the labels
        eps (float): Convergence threshold

    Returns:
        tuple: Two tuples containing (indices, features) for left and right clusters
    """
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    
    # Iterative clustering until convergence
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([
            np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
            np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))
        ]))
    
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])


def main():
    """Main function to run the clustering process."""
    parser = argparse.ArgumentParser(description='Build hierarchical label tree through clustering')
    parser.add_argument('--dataset', type=str, required=False, default='AmazonTitles-670K',
                      help='Dataset name')
    parser.add_argument('--tree', action='store_true',
                      help='Build tree structure')
    parser.add_argument('--id', type=str, required=False, default='0',
                      help='Identifier for the output files')
    
    args = parser.parse_args()
    dataset = args.dataset
    datapath = os.path.join('./Datasets/', dataset)
    
    if dataset in ['WikiSeeAlsoTitles-350K', 'AmazonTitles-670K']:
        final_name = f'label_group_{args.id}'
        final_name = os.path.join(datapath, final_name)
        train_file = os.path.join(datapath, 'bow-train.txt')
        labels_file = os.path.join(datapath, 'bow-labels.txt')
        
        mlb = build_tree_by_level(train_file, labels_file, 1e-4, 15, [], f'{final_name}')
        
        # Convert numeric indices back to original labels
        groups = np.load(f'{final_name}-last.npy', allow_pickle=True)
        new_group = [[mlb.classes_[i] for i in group] for group in groups]
        np.save(f'{final_name}.npy', np.array(new_group))
        
    elif dataset in ['WikiTitles-500K', 'AmazonTitles-3M']:
        final_name = f'label_group_{args.id}'
        final_name = os.path.join(datapath, final_name)
        train_file = os.path.join(datapath, 'bow-train.txt')
        labels_file = os.path.join(datapath, 'bow-labels.txt')
        
        # Different max_leaf values for different datasets
        max_leaf = 10 if dataset == 'WikiTitles-500K' else 30
        mlb = build_tree_by_level(train_file, labels_file, 1e-4, max_leaf, [], f'{final_name}')
        
        # Convert numeric indices back to original labels
        groups = np.load(f'{final_name}-last.npy', allow_pickle=True)
        new_group = [[mlb.classes_[i] for i in group] for group in groups]
        np.save(f'{final_name}.npy', np.array(new_group))


if __name__ == '__main__':
    main()
