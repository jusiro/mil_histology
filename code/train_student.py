import pandas as pd
from mil_data_generator import *
from mil_models_pytorch import*
from mil_trainer_torch import *
from sklearn.utils.class_weight import compute_class_weight
import torch
from torchvision import transforms
import argparse

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(args):

    # INPUTS #
    dir_images = '../data/SICAP_MIL/patches/'
    dir_data_frame = '../data/SICAP_MIL/dataframes/gt_global_slides.xlsx'
    dir_data_frame_test = '../data/SICAP_MIL/dataframes/gt_test_patches.xlsx'
    dir_experiment = '../data/results/' + args.experiment_name + '/'

    classes = ['G3', 'G4', 'G5']
    proportions = ['pG3', 'pG4', 'pG5']
    input_shape = (3, 224, 224)
    images_on_ram = True
    data_augmentation = True
    pMIL = False
    aggregation = 'max'  # 'max', 'mean', 'attentionMIL', 'mcAttentionMIL'
    mode = 'instance'  # 'embedding', 'instance', 'mixed'
    include_background = True
    iterations = 3

    df = pd.read_excel(dir_data_frame)

    metrics = []
    for ii_iteration in np.arange(0, iterations):

        # Set data generators
        dataset_train = MILDataset(dir_images, df[df['Partition'] == 'train'], classes, bag_id='slide_name',
                                   input_shape=input_shape, data_augmentation=False, images_on_ram=images_on_ram,
                                   pMIL=pMIL, proportions=proportions)
        data_generator_train = MILDataGenerator(dataset_train, batch_size=1, shuffle=True, max_instances=512)

        dataset_test = MILDataset(dir_images, df[df['Partition'] == 'test'], classes, bag_id='slide_name',
                                  input_shape=input_shape, data_augmentation=False, images_on_ram=images_on_ram,
                                  pMIL=pMIL, proportions=proportions, dataframe_instances=pd.read_excel(dir_data_frame_test))
        data_generator_test = MILDataGenerator(dataset_test, batch_size=1, shuffle=False, max_instances=512)

        # Test at instance level
        X_test = data_generator_test.dataset.X[data_generator_test.dataset.y_instances[:, 0] != -1, :, :, :]
        Y_test = data_generator_test.dataset.y_instances[data_generator_test.dataset.y_instances[:, 0] != -1, :]

        # Load network
        network = torch.load(dir_experiment + str(ii_iteration) + '_network_weights_best.pth')

        # Pseudolabels on training set
        labels = []
        yhat_one_hot = []
        Yglobal = data_generator_train.dataset.Yglobal
        X = data_generator_train.dataset.X

        for i in np.arange(0, X.shape[0]):
            print(str(i + 1) + '/' + str(X.shape[0]), end='\r')

            # Tensorize input
            x = torch.tensor(X[i, :, :, :]).cuda().float()
            x = x.unsqueeze(0)

            features = network.bb(x)
            yhat = torch.softmax(network.classifier(torch.squeeze(features)), 0)
            yhat = yhat.detach().cpu().numpy()
            yhat_one_hot.append(yhat)

            if np.max(Yglobal[i, 1:]) == 0:
                labels.append(0)
            else:
                if np.argmax(yhat) > 0:
                    if Yglobal[i, np.argmax(yhat)] == 1 and yhat[np.argmax(yhat)] > 0.5:
                        labels.append(np.argmax(yhat))
                    else:
                        labels.append(10)
                else:
                    labels.append(10)
        labels = np.array(labels)
        yhat_one_hot = np.array(yhat_one_hot)

        X = X[labels != 10, :, :, :]
        Y = labels[labels != 10]
        images_id = np.array(dataset_train.images)[labels != 10]
        class_weights = compute_class_weight('balanced', [0, 1, 2, 3], Y)

        # Set student network architecture
        lr = 1e-2
        network = MILArchitecture(classes, mode=mode, aggregation=aggregation, backbone='vgg19',
                                  include_background=include_background).cuda()
        opt = torch.optim.SGD(network.parameters(), lr=lr)

        tranf = torch.nn.Sequential(transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=(-45, 45)),
                                    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                    transforms.ColorJitter(brightness=.5, hue=.3)).cuda()

        training_data = CustomImageDataset(X, Y, transform=False)
        train_dataloader = CustomGenerator(training_data, bs=32, shuffle=True)


        def test_instances(X, Y, network, dir_out, i_iteration):
            network.eval()
            Yhat = []
            for i in np.arange(0, X.shape[0]):
                print(str(i + 1) + '/' + str(X.shape[0]), end='\r')

                # Tensorize input
                x = torch.tensor(X[i, :, :, :]).cuda().float()
                x = x.unsqueeze(0)

                features = network.bb(x)
                yhat = torch.softmax(network.classifier(torch.squeeze(features)), 0)
                yhat = torch.argmax(yhat).detach().cpu().numpy()

                Yhat.append(yhat)

            Yhat = np.array(Yhat)
            Y = np.argmax(Y, 1)

            cr = classification_report(Y, Yhat, target_names=['NC'] + classes, digits=4)
            cm = confusion_matrix(Y, Yhat)
            k2 = cohen_kappa_score(Y, Yhat, weights='quadratic')

            f = open(dir_out + str(i_iteration) + '_report_student.txt', 'w')
            f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
            f.close()

            return k2

        # STUDENT TRAINING


        epochs = 60
        dropout_rate = 0.2
        for i_epoch in np.arange(0, epochs):
            l_epoch = 0

            if (i_epoch + 1) % 25 == 0:
                for g in opt.param_groups:
                    g['lr'] = g['lr'] / 2

            for i_iteration, (X, Y) in enumerate(train_dataloader):
                # Set model to training mode and clear gradients
                network.train()
                opt.zero_grad()

                X = X.cuda().float()
                X = tranf(X)

                logits = network.classifier(torch.nn.Dropout(dropout_rate)(torch.squeeze(network.bb(X))))

                L = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda().float())(logits, Y.type(torch.LongTensor).cuda())

                L.backward()
                opt.step()

                L_iteration = L.detach().cpu().numpy()
                l_epoch += L_iteration

                info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.6f}".format(
                    i_epoch + 1, epochs, i_iteration + 1, len(train_dataloader), L_iteration)
                print(info, end='\r')

            l_epoch = l_epoch/len(train_dataloader)

            k2 = test_instances(X_test, Y_test, network, dir_experiment, ii_iteration)

            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.6f}; k2={:.6f}".format(
                i_epoch+1, epochs, i_iteration, len(train_dataloader), l_epoch, k2)
            print(info, end='\n')

        torch.save(network, dir_experiment + str(ii_iteration) + '_student_network_weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="test_test_test", type=str)

    args = parser.parse_args()
    main(args)
