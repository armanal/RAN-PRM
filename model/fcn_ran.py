### Fully Convolutional Residual Attention Network ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_attention_network import ResidualAttentionModel_448input


# class FCN_RAN(nn.Module):
#
#     def __init__(self, num_classes):
#         super(FCN_RAN, self).__init__()
#
#         model = ResidualAttentionModel_448input()
#         # feature encoding
#         self.features = nn.Sequential(
#             model.conv1,
#             model.mpool1,
#             model.residual_block0,
#             model.attention_module0,
#             model.residual_block1,
#             model.attention_module1,
#             model.residual_block2,
#             model.attention_module2,
#             model.attention_module2_2,
#             model.residual_block3,
#             model.attention_module3,
#             model.attention_module3_2,
#             model.attention_module3_3,
#             model.residual_block4,
#             model.residual_block5,
#             model.residual_block6) #,
#             # model.mpool2)
#
#         # classifier
#         fake_input = torch.zeros((3,448,448)).unsqueeze_(dim=0)
#         out = self.features(fake_input)
#         num_features = out.size()[1]
#         self.classifier = nn.Sequential(
#             nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


class FCN_RAN(nn.Module):

    def __init__(self, num_classes):
        super(FCN_RAN, self).__init__()

        model = ResidualAttentionModel_448input()
        # feature encoding
        self.attention = [model.attention_module0,
                        model.attention_module2,
                        model.attention_module2_2,
                        model.attention_module3,
                        model.attention_module3_2,
                        model.attention_module3_3]
                        
        self.features = nn.Sequential(
            model.conv1,
            model.mpool1,
            model.residual_block0,
            self.attention[0],
            model.residual_block1,
            model.attention_module1,
            model.residual_block2,
            self.attention[1],
            self.attention[2],
            model.residual_block3,
            self.attention[3],
            self.attention[4],
            self.attention[5],
            model.residual_block4,
            model.residual_block5,
            model.residual_block6) #,
            # model.mpool2)

        # classifier
        fake_input = torch.zeros((3,448,448)).unsqueeze_(dim=0)
        out = self.features(fake_input)
        num_features = out.size()[1]
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
