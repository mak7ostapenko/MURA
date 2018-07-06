from keras.preprocessing.image import ImageDataGenerator


def get_datagen(params):
    """
    Create data generator

    Arguments:
        params - dictionary of parameters for setting up generator

    Returns:
        data generator from directory

    """
    data_gen_param = params.datagen
    gen_param = params.generator
    data_gen = ImageDataGenerator(**data_gen_param)
    generator = data_gen.flow_from_directory(**gen_param)

    return generator


def class_weights(labels):
    """
    Compute balanced class weights

    Arguments:
        labels - labels of training samples

    Returns:
        Dictionary with weight for each class label

    """
    unique, counts = np.unique(labels, return_counts=True)
    weights = counts / np.sum(counts)
    weights_dict = dict(zip(unique, weights))

    return weights_dict

