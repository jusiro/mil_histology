import pandas as pd
from mil_data_generator import *
from mil_models_pytorch import*
from mil_trainer_torch import *
import argparse

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(args):

    metrics = []
    for i_iteration in np.arange(0, args.iterations):
        id = str(i_iteration) + '_'
        df = pd.read_excel(args.dir_data_frame)

        # Set data generators
        dataset_train = MILDataset(args.dir_images, df[df['Partition'] == 'train'], args.classes,
                                   bag_id='slide_name', input_shape=args.input_shape,
                                   data_augmentation=args.data_augmentation, images_on_ram=args.images_on_ram,
                                   pMIL=args.pMIL, proportions=args.proportions)
        data_generator_train = MILDataGenerator(dataset_train, batch_size=1, shuffle=True, max_instances=512)

        dataset_val = MILDataset(args.dir_images, df[df['Partition'] == 'val'], args.classes,
                                 bag_id='slide_name', input_shape=args.input_shape,
                                 data_augmentation=args.data_augmentation, images_on_ram=args.images_on_ram,
                                 pMIL=args.pMIL, proportions=args.proportions)
        data_generator_val = MILDataGenerator(dataset_val, batch_size=1, shuffle=False, max_instances=512)

        dataset_test = MILDataset(args.dir_images, df[df['Partition'] == 'test'], args.classes,
                                  bag_id='slide_name', input_shape=args.input_shape,
                                  data_augmentation=args.data_augmentation, images_on_ram=args.images_on_ram,
                                  pMIL=args.pMIL, proportions=args.proportions,
                                  dataframe_instances=pd.read_excel(args.dir_data_frame_test))
        data_generator_test = MILDataGenerator(dataset_test, batch_size=1, shuffle=False, max_instances=512)

        # Set network architecture
        network = MILArchitecture(args.classes, mode=args.mode, aggregation=args.aggregation,
                                  backbone='vgg19', include_background=args.include_background)

        # Perform training
        trainer = MILTrainer(args.dir_results + args.experiment_name + '/', network,
                             lr=args.lr, pMIL=args.pMIL, margin=args.margin,
                             alpha_ic=args.alpha_ic, alpha_pc=args.alpha_pc, t_ic=args.t_ic,
                             t_pc=args.t_pc, alpha_ce=args.alpha_ce, id=id,
                             early_stopping=args.early_stopping, scheduler=args.scheduler,
                             virtual_batch_size=args.virtual_batch_size,
                             criterion=args.criterion,
                             alpha_H=args.alpha_H)
        trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                      test_generator=data_generator_test, epochs=args.epochs)

        metrics.append([list(trainer.metrics.values())[1:]])

    # Get overall metrics
    metrics = np.squeeze(np.array(metrics))

    mu = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)

    info = "AUCtest={:.4f}({:.4f}) ; AUCval={:.4f}({:.4f})  ; acc={:.4f}({:.4f}) ; f1-score={:.4f}({:.4f}) ; k2={:.4f}({:.4f})".format(
          mu[0], std[0], mu[1], std[1], mu[2], std[2], mu[3], std[3], mu[4], std[4])
    if args.alpha_pc > 0:
        info += " ; constrain_cumpliment={:.4f}({:.4f}) ; constrain_proportion={:.4f}({:.4f})".format(
          mu[5], std[5], mu[6], std[6])

    f = open(args.dir_results + args.experiment_name + '/' + 'method_metrics.txt', 'w')
    f.write(info)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_images", default='../data/SICAP_MIL/patches/', type=str)
    parser.add_argument("--dir_data_frame", default='../data/SICAP_MIL/dataframes/gt_global_slides.xlsx', type=str)
    parser.add_argument("--dir_data_frame_test", default='../data/SICAP_MIL/dataframes/gt_test_patches.xlsx', type=str)
    parser.add_argument("--dir_results", default='../data/results/', type=str)
    parser.add_argument("--criterion", default='z', type=str)
    parser.add_argument("--experiment_name", default="test_test_test", type=str)
    parser.add_argument("--classes", default=['G3', 'G4', 'G5'], type=list)
    parser.add_argument("--proportions", default=['Primary', 'Secondary'], type=list)
    parser.add_argument("--input_shape", default=(3, 224, 224), type=list)
    parser.add_argument("--images_on_ram", default=False, type=bool)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--aggregation", default="max", type=str)
    parser.add_argument("--mode", default="instance", type=str)
    parser.add_argument("--include_background", default=True, type=bool)
    parser.add_argument("--lr", default=1*1e-2, type=float)
    parser.add_argument("--pMIL", default=False, type=bool)

    parser.add_argument("--alpha_ce", default=1., type=float)
    parser.add_argument("--margin", default=0., type=float)
    parser.add_argument("--alpha_ic", default=1, type=float)
    parser.add_argument("--alpha_pc", default=1, type=float)
    parser.add_argument("--alpha_H", default=0, type=float)
    parser.add_argument("--t_ic", default=15, type=float)
    parser.add_argument("--t_pc", default=5, type=float)
    parser.add_argument("--data_augmentation", default=True, type=bool)
    parser.add_argument("--iterations", default=3, type=int)

    parser.add_argument("--early_stopping", default=True, type=bool)
    parser.add_argument("--scheduler", default=True, type=bool)

    parser.add_argument("--virtual_batch_size", default=1, type=int)

    args = parser.parse_args()
    main(args)

