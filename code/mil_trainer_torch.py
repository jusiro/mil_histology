import sklearn
import torch
import numpy as np
import datetime
import cv2
import sklearn.metrics
import os
from timeit import default_timer as timer
import json
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from losses import *
from sklearn.cluster import KMeans
import torch.nn.functional as F
import random

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class MILTrainer():
    def __init__(self, dir_out, network, lr=1*1e-4, pMIL=False, margin=0, t_ic=10,
                 t_pc=10, alpha_ic=1, alpha_pc=1, alpha_ce=1, id='', early_stopping=False,
                 scheduler=False, virtual_batch_size=1, criterion='auc', alpha_H=0.01):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.test_generator = []
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []
        self.pMIL = pMIL
        self.alpha_ce = alpha_ce
        self.best_criterion = 0
        self.best_epoch = 0
        self.metrics = {}
        self.id = id
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.constrain_cumpliment_lc = []
        self.constrain_proportion_lc = []
        self.criterion = criterion
        self.alpha_H = alpha_H
        self.H_iteration = 0.
        self.H_epoch = 0.

        # Set optimizers
        self.params = list(self.network.parameters())

        if self.pMIL:
            self.Lp_iteration = 0
            self.Lp_epoch = 0
            self.Lp_lc = []
            self.m = margin
            self.t_ic = t_ic
            self.t_pc = t_pc
            self.alpha_ic = alpha_ic
            self.alpha_pc = alpha_pc
            self.constrain_cumpliment = 0.
            self.constraint_proportion = 0.


        self.opt = torch.optim.SGD(self.params, lr=self.lr)
        #self.opt = torch.optim.Adam(self.params, lr=self.lr)

        # Set losses
        if network.mode == 'embedding' or network.mode == 'mixed':
            self.L = torch.nn.BCEWithLogitsLoss().cuda()
        elif network.mode == 'instance':
            self.L = torch.nn.BCELoss().cuda()

    def train(self, train_generator, val_generator, test_generator, epochs):
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.preds_train = []
        self.refs_train = []

        # Move network to gpu
        self.network.cuda()

        self.init_time = timer()
        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch losses
            self.L_epoch = 0
            self.Lpc_iteration = 0
            self.Lic_iteration = 0
            self.Lic_epoch = 0
            self.Lpc_epoch = 0
            self.H_iteration = 0.
            self.H_epoch = 0.
            self.constrain_cumpliment_iteration = 0.
            self.constrain_cumpliment_epoch = 0.
            self.constrain_proportion_epoch = 0.
            self.constrain_ic_proportion_epoch = 0.
            self.j = 0.
            self.jj = 0.
            n = 0
            nn = 0

            if self.scheduler:
                if (self.i_epoch + 1) % 50 == 0:
                    for g in self.opt.param_groups:
                        g['lr'] = self.lr / 2

            # Loop over training dataset
            print('[Training]: at bag level...')
            for self.i_iteration, (X, Y, O, X_augm) in enumerate(self.train_generator):

                X = torch.tensor(X).cuda().float()
                if X_augm is None:
                    X_augm = X
                else:
                    X_augm = torch.tensor(X_augm).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # Set model to training mode and clear gradients
                self.network.train()

                # Forward network
                Yhat, yhat, features = self.network(X_augm)

                if self.network.mode == 'instance':
                    Yhat = torch.clip(Yhat, min=0.01, max=0.98)

                # Estimate losses
                Lce = self.L(Yhat, torch.squeeze(Y))

                # Update overall losses
                L = Lce * self.alpha_ce

                if self.alpha_H > 0:
                    H = torch.mean(-torch.sum(yhat * torch.log(yhat + 1e-12), dim=(-1)))
                    self.H_iteration = H

                    L += - self.alpha_H * self.H_iteration

                if self.pMIL:

                    O_ic = np.array(O[0]).astype('float32')
                    O_pc = np.array(O[1]).astype('float32')

                    if self.network.include_background:
                        yhat = yhat[:, 1:]

                    if self.alpha_ic > 0:
                        if np.max(np.abs(O_ic)) == 1:
                            # Move O matrix to tensor
                            Ot = torch.tensor(O_ic).cuda().float()
                            # Obtain proportion vector
                            P = torch.mean(yhat, 0)
                            # Obtain z
                            z = torch.matmul(Ot, P.unsqueeze(-1))
                            self.Lic_iteration = log_barrier(z, t=self.t_ic).squeeze()
                            # Update overall losses
                            L += self.alpha_ic * self.Lp_iteration
                            n += 1

                            self.constrain_ic_proportion_epoch += np.sum(z.cpu().detach().numpy())
                            self.jj += 1
                        else:
                            self.Lic_iteration = torch.tensor(0).cuda()

                    if self.alpha_pc > 0:
                        if np.max(np.abs(O_pc)) == 1:
                            # Move O matrix to tensor
                            Ot2 = torch.tensor(O_pc).cuda().float()
                            # Obtain proportion vector
                            P = torch.mean(yhat, 0)
                            # Obtain z
                            z = torch.matmul(Ot2, P.unsqueeze(-1)) + self.m
                            self.Lpc_iteration = log_barrier(z, t=self.t_pc).squeeze()
                            # Update overall losses
                            L += self.alpha_pc * self.Lpc_iteration
                            nn += 1

                            if (z-self.m) < 0:
                                self.constrain_cumpliment_iteration = 1
                            else:
                                self.constrain_cumpliment_iteration = 0
                            self.constrain_proportion_epoch += z.cpu().detach().numpy() - self.m
                            self.j += 1
                        else:
                            self.Lpc_iteration = torch.tensor(0).cuda()
                            self.constrain_cumpliment_iteration = 0

                # Backward gradients
                L = L / self.virtual_batch_size
                L.backward()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                ######################################
                ## --- Iteration/Epoch end

                # Save predictions
                self.preds_train.append(Yhat.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Display losses per iteration
                self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, self.iterations,
                                    Lce.cpu().detach().numpy(),
                                    end_line='\r')

                # Update epoch's losses
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)
                if self.pMIL and np.max(np.abs(O_ic)) == 1 and self.alpha_ic > 0:
                    self.Lic_epoch += self.Lic_iteration.cpu().detach().numpy()
                if self.pMIL and np.max(np.abs(O_pc)) == 1 and self.alpha_pc > 0:
                    self.Lpc_epoch += self.Lpc_iteration.cpu().detach().numpy()
                    self.constrain_cumpliment_epoch += self.constrain_cumpliment_iteration
                if self.alpha_H > 0.:
                    self.H_epoch += self.H_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            if self.pMIL and self.alpha_ic > 0:
                self.Lic_epoch = self.Lic_epoch / n
                self.constrain_ic_proportion_epoch = np.squeeze(self.constrain_ic_proportion_epoch) / self.jj
            if self.pMIL and self.alpha_pc > 0:
                self.Lpc_epoch = self.Lpc_epoch / nn
                self.constrain_cumpliment_epoch = self.constrain_cumpliment_epoch / nn
                self.constrain_proportion_epoch = np.squeeze(self.constrain_proportion_epoch) / self.j

                self.constrain_cumpliment_lc.append(self.constrain_cumpliment_epoch)
                self.constrain_proportion_lc.append(self.constrain_proportion_epoch)

            self.on_epoch_end()

            if self.early_stopping:
                if self.i_epoch + 1 == (self.best_epoch + 20):
                    break

    def on_epoch_end(self):

        # Obtain epoch-level metrics
        macro_auc = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.array(self.preds_train), multi_class='ovr')
        self.macro_auc_lc_train.append(macro_auc)

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc,
                            end_line='\n')
        # Update learning curves
        self.L_lc.append(self.L_epoch)

        # Obtain results on validation set
        Lce_val, macro_auc_val = self.test_bag_level_classification(self.val_generator)

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)

        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4)}
        with open(self.dir_results + self.id + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        if (self.i_epoch + 1) > 10:
            if self.criterion == 'auc':
                if self.best_criterion < self.macro_auc_lc_val[-1]:
                    self.best_criterion = self.macro_auc_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)

                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

            elif self.criterion == 'z':
                if self.best_criterion < (-self.constrain_proportion_epoch):
                    self.best_criterion = -self.constrain_proportion_epoch
                    self.best_epoch = (self.i_epoch + 1)

                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

        # Each xx epochs, test models and plot learning curves
        if (self.i_epoch + 1) % 5 == 0:
            # Save weights
            torch.save(self.network, self.dir_results + self.id + 'network_weights.pth')

            # Plot learning curve
            self.plot_learning_curves()

            # Test at instance level
            X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, 0] != -1, :, :, :]
            Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:, 0] != -1, :]
            acc, f1, k2 = self.test_instance_level_classification(X, Y, self.test_generator.dataset.classes)

        if (self.epochs == (self.i_epoch + 1)) or (self.early_stopping and (self.i_epoch + 1 == (self.best_epoch + 20))):
            print('-' * 20)
            print('-' * 20)

            self.network = torch.load(self.dir_results + self.id + 'network_weights_best.pth')

            # Obtain results on validation set
            Lce_val, macro_auc_val = self.test_bag_level_classification(self.val_generator)

            # Obtain results on validation set
            Lce_test, macro_auc_test = self.test_bag_level_classification(self.test_generator)

            # Test at instance level
            X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, 0] != -1, :, :, :]
            Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:, 0] != -1, :]
            acc, f1, k2 = self.test_instance_level_classification(X, Y, self.test_generator.dataset.classes)

            metrics = {'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4),
                       'AUCval': np.round(macro_auc_val, 4), 'acc': np.round(acc, 4),
                       'f1': np.round(f1, 4), 'k2': np.round(k2, 4),
                       }

            if self.alpha_pc:
                metrics['constrain_cumpliment'] = np.round(self.constrain_cumpliment_lc[self.best_epoch-1], 4)
                metrics['constrain_proportion'] = np.round(self.constrain_proportion_lc[self.best_epoch-1], 4)

            with open(self.dir_results + self.id + 'best_metrics.json', 'w') as fp:
                json.dump(metrics, fp)
            print(metrics)

            self.metrics = metrics
            print('-' * 20)
            print('-' * 20)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            for i in range(x.shape[0]):
                axes.plot(x[i, :], y[i, :], 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 1, figsize=(20, 15))
        plot_subplot(axes[0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.L_lc, self.Lce_lc_val]), "Lce")
        plot_subplot(axes[1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.macro_auc_lc_train, self.macro_auc_lc_val]), "mAUC")

        plt.savefig(self.dir_results + self.id + 'learning_curve.png')

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc=0, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f}".format(
                i_epoch, epochs, iteration, total_iterations, Lce, macro_auc)

        if self.alpha_H > 0:
            if end_line == '\n':
                info += ' ; H=' + str(np.round(self.H_epoch, 4))
            else:
                info += ' ; H=' + str(np.round(self.H_iteration.cpu().detach().numpy(), 4))

        if self.pMIL and end_line == '\n':
            if self.alpha_pc > 0:
                info += ' ; IC=' + str(np.round(self.Lic_epoch, 4))
                info += '{' + str(np.round(self.constrain_ic_proportion_epoch, 4)) + '}'
            if self.alpha_ic > 0:
                info += ' ; PC=' + str(np.round(self.Lpc_epoch, 4))
                info += '{' + str(np.round(self.constrain_cumpliment_epoch, 4)) + '}'
                info += '{' + str(np.round(self.constrain_proportion_epoch, 4)) + '}  '
        if self.pMIL and end_line == '\r':
            if self.alpha_pc > 0:
                info += ' ; IC=' + str(np.round(self.Lic_iteration.cpu().detach().numpy(), 4))
            if self.alpha_ic > 0:
                info += ' ; PC=' + str(np.round(self.Lpc_iteration.cpu().detach().numpy(), 4))
                info += '{' + str(np.round(self.constrain_cumpliment_iteration, 4)) + '}  '

        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def test_instance_level_classification(self, X, Y, classes):
        classes = ['NC'] + classes

        self.network.eval()
        print(['INFO: Testing at instance level...'])

        Yhat = []
        for iInstance in np.arange(0, X.shape[0]):
            print(str(iInstance+1) + '/' + str(X.shape[0]), end='\r')

            # Tensorize input
            x = torch.tensor(X[iInstance, :, :, :]).cuda().float()
            x = x.unsqueeze(0)

            if self.network.aggregation == 'mcAttentionMIL':
                yhat = self.network.milAggregation(torch.squeeze(self.network.bb(x)).unsqueeze(0))[1]
                yhat = torch.argmax(yhat).detach().cpu().numpy()
            else:
                # Make prediction
                if not self.network.prototypical:
                    yhat = torch.softmax(self.network.classifier(torch.squeeze(self.network.bb(x))), 0)
                else:
                    yhat = torch.softmax(- torch.cdist(torch.squeeze(self.network.bb(x)).unsqueeze(0), self.network.C, p=2.0), 1)
                yhat = torch.argmax(yhat).detach().cpu().numpy()

            Yhat.append(yhat)
        Yhat = np.array(Yhat)
        Y = np.argmax(Y, 1)

        cr = classification_report(Y, Yhat, target_names=classes, digits=4)
        acc = accuracy_score(Y, Yhat)
        f1 = f1_score(Y, Yhat, average='macro')
        cm = confusion_matrix(Y, Yhat)
        k2 = cohen_kappa_score(Y, Yhat, weights='quadratic')

        print('Instance Level kappa: ' + str(np.round(k2, 4)), end='\n')

        f = open(self.dir_results + self.id + 'report.txt', 'w')
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
        f.close()

        return acc, f1, k2

    def test_bag_level_classification(self, test_generator, binary=False):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (X, Y, O, _) in enumerate(test_generator):
            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()

            # Set model to training mode and clear gradients

            # Forward network
            Yhat, _, _ = self.network(X)
            # Estimate losses
            Lce = self.L(Yhat, torch.squeeze(Y))
            Lce_e += Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(Yhat.detach().cpu().numpy())

            # Display losses per iteration
            self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                                Lce.cpu().detach().numpy(),
                                end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if binary:
            Yhat_all = np.max(Yhat_all, 1)
            Y_all = np.max(Y_all, 1)

        macro_auc = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                            Lce_e, macro_auc,
                            end_line='\n')

        return Lce_e, macro_auc
