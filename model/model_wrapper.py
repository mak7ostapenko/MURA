from torch import nn


class Net(nn.Module):
    def __init__(self, base_model, name, input_channels, pretrained=True, num_classes=2):
        super(Net, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.base_model = base_model(pretrained=pretrained)

        if self.name == 'resnet':
            self.base_model.fc = nn.Linear(2048, self.num_classes)
        elif self.name == 'densenet121':
            self.base_model.classifier = nn.Linear(1024, self.num_classes)
        elif self.name =='densenet169':
            self.base_model.classifier = nn.Linear(1664, self,num_classes)
        else:
            raise ValueError('Undefined base model name.')

    def forward(self, input):
        """Build and prepare a model that will be ready for training

        Arguments
            input : tensor of shape = (num_samples, height, width, channels)
                Batch of images

        Returns
            output : array
                Final output of model after forward pass

        """
        output = self.base_model(input)

        return output





















