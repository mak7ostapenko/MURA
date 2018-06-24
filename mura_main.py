from easydict import EasyDict
from model.models import ModelFactory


farm = ModelFactory(model_names=['resnet', 'densenet'], num_clasess=2)

model_parameters = EasyDict({'resnet':
                                 {'weight': 'imagenet',
                                  'widht': 224, 'height': 224,
                                  'include_top': False},
                             'densenet':
                                 {'weight': 'imagenet',
                                  'widht': 224, 'height': 224,
                                  'include_top': False},
                             })

model_dict = farm.get_model_dict(model_parameters)


