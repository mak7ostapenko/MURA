import torch
import numpy as np
import pandas as pd


def get_class_weights(labels):
    """Compute balanced class weights

    Arguments
        labels : list ar array
            Labels of training samples

    Returns
        weights_dict : dict
              Weight for each class label

    """
    unique, counts = np.unique(labels, return_counts=True)
    weights = counts / np.sum(counts)
    weights_dict = dict(zip(unique, weights))
    return weights_dict


def transform_csv(path_to_csv, new_path):
    """Add columns input input DataFrame and save in new file

    Arguments
        path_to_csv : str
            Path to csv file with labels and path to images
        new_path : str
            Path to place where you want to save expanded csv file

    """
    train_data = pd.read_csv(path_to_csv)
    # create new dataset frame and add columns
    expanded_frame = pd.DataFrame()
    expanded_frame['path'] = train_data['img']
    expanded_frame['label'] = train_data['label']
    type_pat = np.array([train_data['img'].apply(lambda path: path.split('/')[2:4])])
    type_pat = type_pat[0, :, :]
    expanded_frame['study_type'], expanded_frame['patient'] = type_pat[:, 0], type_pat[:, 1]

    expanded_frame.to_csv(new_path)


def save_checkpoint(model, optimizer, checkpoint_path='./model_checkpoints/model.pth'):
    """Save weight of model"""
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('Model saved to {}'.format(checkpoint_path))


def load_checkpoint(model, optimizer, checkpoint_path='./model_checkpoints/model.pth'):
    """Download weight of model"""
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizrieer'])
    print('Model are loaded from {}'.format(checkpoint_path))