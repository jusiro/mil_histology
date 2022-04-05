import torch
import torchvision
import numpy as np


class MILArchitecture(torch.nn.Module):

    def __init__(self, classes, mode='embedding', aggregation='mean', backbone='VGG19', include_background=False):
        super(MILArchitecture, self).__init__()

        """Data Generator object for MIL.
            CNN based architecture for MIL classification.
        Args:
          classes: 
          mode:
          aggregation: max, mean, attentionMIL, mcAttentionMIL
          backbone:
          include_background:

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'

        self.classes = classes
        self.mode = mode
        self.aggregation = aggregation
        self.backbone = backbone
        self.include_background = include_background
        self.C = []
        self.prototypical = False

        if self.include_background:
            self.nClasses = len(classes) + 1
        else:
            self.nClasses = len(classes)
        self.eps = 1e-6

        # Backbone
        self.bb = Encoder(pretrained=True, backbone=backbone, aggregation=True)
        # Classifiers
        if self.aggregation == 'mcAttentionMIL':
            self.classifiers = torch.nn.ModuleList()
            for i in np.arange(0, self.nClasses):
                self.classifiers.append(torch.nn.Linear(512, 1))
        else:
            self.classifier = torch.nn.Linear(512, self.nClasses)
        # MIL aggregation
        self.milAggregation = MILAggregation(aggregation=aggregation, nClasses=self.nClasses, mode=self.mode)

    def forward(self, images):
        # Patch-Level feature extraction
        features = self.bb(images)

        if self.mode == 'instance':
            # Classification
            patch_classification = torch.softmax(self.classifier(torch.squeeze(features)), 1)

            # MIL aggregation
            global_classification = self.milAggregation(patch_classification)

        if self.mode == 'embedding' or self.mode == 'mixed':  # Activation on BCE loss
            # Embedding aggregation
            if self.aggregation == 'mcAttentionMIL':
                embedding, patch_classification = self.milAggregation(torch.squeeze(features))
                global_classifications = []
                for i in np.arange(0, self.nClasses):
                    global_classifications.append(self.classifiers[i](embedding[:, i]))
                global_classification = torch.cat(global_classifications, dim=0)
            elif self.aggregation == 'attentionMIL':
                embedding, w = self.milAggregation(torch.squeeze(features))
                global_classification = self.classifier(embedding)
                patch_classification = w
            else:
                embedding = self.milAggregation(torch.squeeze(features))
                global_classification = self.classifier(embedding)
                patch_classification = self.classifier(torch.squeeze(features))

        if self.include_background:
            global_classification = global_classification[1:]

        return global_classification, patch_classification, features


class Encoder(torch.nn.Module):

    def __init__(self, pretrained=True, backbone='resnet18', aggregation=False):
        super(Encoder, self).__init__()

        self.aggregation = aggregation
        self.pretrained = pretrained
        self.backbone = backbone

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
            self.F = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
        elif backbone == 'vgg19':
            vgg19 = torchvision.models.vgg16(pretrained=pretrained)
            self.F = vgg19.features

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        out = self.F(x)

        # register the hook
        h = out.register_hook(self.activations_hook)

        if self.aggregation:
            out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)

        return out

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad


class MILAggregation(torch.nn.Module):
    def __init__(self, aggregation='mean', nClasses=2, mode='embedding'):
        super(MILAggregation, self).__init__()

        """Aggregation module for MIL.
        Args:
          aggregation:

        Returns:
          MILAggregation module for CNN MIL Architecture
        Last Updates: Julio Silva (19/03/21)
        """

        self.mode = mode
        self.aggregation = aggregation
        self.nClasses = nClasses

        if self.aggregation == 'attentionMIL':
            self.attentionModule = attentionMIL()

        if self.aggregation == 'mcAttentionMIL':
            self.attentionModules = torch.nn.ModuleList()
            for i in np.arange(0, self.nClasses):
                self.attentionModules.append(attentionMIL())

    def forward(self, feats):

        if self.aggregation == 'max':
            embedding = torch.max(feats, dim=0)[0]
            return embedding
        elif self.aggregation == 'mean':
            embedding = torch.mean(feats, dim=0)
            return embedding
        elif self.aggregation == 'attentionMIL':
            # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario at instance-level
            embedding, w_logits = self.attentionModule(feats)
            return embedding, torch.softmax(w_logits, dim=0)

        elif self.aggregation == 'mcAttentionMIL':
            attention_weights = []
            embeddings = []
            for i in np.arange(0, self.nClasses):
                embeddings.append(self.attentionModules[i](feats)[0].unsqueeze(1))
                attention_weights.append(self.attentionModules[i](feats)[1])
            #patch_classification = torch.softmax(torch.cat(attention_weights, 1), 0)
            if self.mode == 'embedding':
                embedding = torch.cat(embeddings, 1)
                patch_classification = torch.softmax(torch.cat(attention_weights, 1), 1)

                #w = patch_classification

            elif self.mode == 'mixed':

                patch_classification = torch.softmax(torch.cat(attention_weights, 1), 0)
                w = patch_classification * (1/torch.sum(patch_classification, 0) + 1e-6)

                feats = torch.transpose(feats, 1, 0)
                embedding = torch.squeeze(torch.mm(feats, w))

            return embedding, patch_classification


class attentionMIL(torch.nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(attentionMIL, self).__init__()

        # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario.

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )
        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, feats):
        # Attention weights computation
        A_V = self.attention_V(feats)  # Attention
        A_U = self.attention_U(feats)  # Gate
        w_logits = self.attention_weights(A_V * A_U)  # Probabilities - softmax over instances

        # Weighted average computation per class
        feats = torch.transpose(feats, 1, 0)
        embedding = torch.squeeze(torch.mm(feats, torch.softmax(w_logits, dim=0)))  # KxL

        return embedding, w_logits