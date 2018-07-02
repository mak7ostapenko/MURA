from easydict import EasyDict
from model.models import ModelFactory


farm = ModelFactory(model_names=['resnet', 'densenet'], num_clasess=2)

model_parameters = EasyDict({'resnet':
                                 {'weights': 'imagenet',
                                  'widht': 224, 'height': 224,
                                  'include_top': False},
                             'densenet':
                                 {'weights': 'imagenet',
                                  'widht': 224, 'height': 224,
                                  'include_top': False},
                             })

model_dict = farm.get_model_dict(model_parameters)

# train the factory or a single model ?
# must have ability  to do first and second!!!
# so in the class model train a single model and in the class factory train several models

train_param = EasyDict({'datagenerator':
    {
        'train':
            {
                'datagen':
                    {
                        'rescale': 1. / 255,
                        'rotation_range': 45,
                        'width_shift_range': 0.2,
                        'height_shift_range': 0.2,
                        'zoom_range': 0.2,
                        'horizontal_flip': True,
                    },
                'generator':
                    {
                        'directory': 'data/train',
                        'shuffle': True,
                        "target_size": (224, 224),
                        "class_mode": 'binary',
                        "batch_size": 64,
                    },

            },
        'val':
            {
                'datagen':
                    {
                        'rescale': 1. / 255,
                    },
                'generator':
                    {
                        'directory': 'data/val',
                        'shuffle': True,
                        "target_size": (224, 224),
                        "class_mode": 'binary',
                        "batch_size": 64,
                    },
            },
    },
})

farm.train_factory(model_dict, train_param)
