import pandas as pd
from mil_data_generator import *
from mil_models_pytorch import*
from mil_trainer_torch import *
from sklearn.utils.class_weight import compute_class_weight
import torch
from torchvision import transforms
from PIL import Image
import argparse

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(args):

    def image_normalization(x, input_shape, channel_first=True):
        # image resize
        x = cv2.resize(x, (input_shape[1], input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    # INPUTS #
    dir_slides = '../data/SICAP_MIL/slides/'
    dir_data_frame = '../data/SICAP_MIL/dataframes/gt_global_slides.xlsx'
    dir_experiment = '../data/results/' + args.experiment_name + '/'

    classes = ['G3', 'G4', 'G5']
    input_shape = (3, 224, 224)
    images_on_ram = True
    patch_size = 512
    overlap = 0.25
    save_annotations = args.save_annotations

    # Load df and take only test biopsies
    df = pd.read_excel(dir_data_frame)
    df = df[df['Partition'] == 'test']
    slices_test = list(df['slide_name'])

    # slices in folder
    slices = os.listdir(dir_slides)

    # Load network -- we use the first iteration model as example
    network = torch.load(dir_experiment + str(0) + '_network_weights_best.pth').cuda()

    if not os.path.isdir(dir_experiment + '/visualizations/'):
        os.mkdir(dir_experiment + '/visualizations/')
    if save_annotations:
        if not os.path.isdir(dir_experiment + '/visualizations_gt/'):
            os.mkdir(dir_experiment + '/visualizations_gt/')

    c = 0
    for iSlide in slices:
        c += 1
        print(str(c) + '/' + str(len(slices)))

        if iSlide.split('_')[0] in slices_test:

            wsi = Image.open(os.path.join(dir_slides, iSlide))
            wsi = np.asarray(wsi)

            if save_annotations:
                if os.path.isfile(os.path.join('../data/SICAP_MIL/annotation_masks/', iSlide)):
                    wsi_gt = Image.open(os.path.join('../data/SICAP_MIL/annotation_masks/', iSlide))
                    wsi_gt = np.asarray(wsi_gt)

            tissue = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(tissue, 120, 255, cv2.THRESH_BINARY +
                                         cv2.THRESH_OTSU)
            tissue = tissue < (ret)
            tissue = cv2.morphologyEx(np.uint8(tissue*255), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) / 255

            if not save_annotations:
                output = np.zeros((wsi.shape[0], wsi.shape[1], 4))
                npatches = np.zeros((wsi.shape[0], wsi.shape[1]))
                x0 = 0
                while (x0 + patch_size) <= wsi.shape[1]:
                    y0 = 0
                    while (y0 + patch_size) <= wsi.shape[0]:
                        # If there is tissue, get predictions
                        if np.mean(tissue[y0:y0+patch_size, x0:x0+patch_size]) > 0.2:
                            # Take patch
                            patch = wsi[y0:y0+patch_size, x0:x0+patch_size, :]
                            # Pre-process patch
                            x = image_normalization(patch.copy(), input_shape)
                            x = torch.tensor(x).cuda().float().unsqueeze(0)
                            # Forward
                            features = network.bb(x)
                            yhat = torch.softmax(network.classifier(torch.squeeze(features)), 0)
                            yhat = yhat.detach().cpu().numpy()
                            # Update visualization heatmap
                            output[y0:y0+patch_size, x0:x0+patch_size, :] += yhat
                            npatches[y0:y0+patch_size, x0:x0+patch_size] += 1

                        y0 = int(y0 + patch_size*overlap)
                    x0 = int(x0 + patch_size*overlap)

                a = output / (np.expand_dims(npatches, -1) + 1e-6)
                mask = np.argmax(a, axis=-1)
                mask = mask * tissue

                colors = np.float64(np.concatenate([np.expand_dims(mask == 3, -1),
                                                    np.expand_dims(mask == 1, -1),
                                                    np.expand_dims(mask == 2, -1)], axis=-1))
                overlay = wsi + 0.3 * (colors * 254)
                overlay = np.clip(overlay, 0, 254) / 255

                im = Image.fromarray((overlay * 255).astype(np.uint8))
                im.save(dir_experiment + '/visualizations/' + iSlide)

            if save_annotations and os.path.isfile(os.path.join('../data/SICAP_MIL/annotation_masks/', iSlide)):
                colors = np.float64(np.concatenate([np.expand_dims(wsi_gt >= 170, -1),
                                                    np.expand_dims((wsi_gt >= 25) * (wsi_gt <= 80) , -1),
                                                    np.expand_dims((wsi_gt >= 80) * (wsi_gt <= 170), -1)], axis=-1))
                overlay = wsi + 0.3 * (colors * 254)
                overlay = np.clip(overlay, 0, 254) / 255

                im = Image.fromarray((overlay * 255).astype(np.uint8))
                im.save(dir_experiment + '/visualizations_gt/' + iSlide)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="test_test_test", type=str)
    parser.add_argument("--save_annotations", default=False, type=bool)

    args = parser.parse_args()
    main(args)
