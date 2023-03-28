# Weakly Supervised Classification of Histology WSIs 
This repository contains open code for weakly supervised histology whole slide image classification under the paradigm of Multiple Instance Learning (MIL). Also, it containts the link to SICAP-MIL dataset, a publicly available dataset of prostate whole slide images.

## SICAP-MIL Dataset

SICAP-MIL dataset is composed of 350 prostate Whole Slide Images (WSIs) at 10x resolution. The slides contain the WSI-level Gleason score, including both primary and secondary Gleason grades. Also, we share the post-processed version of the dataset, with patches extracted using non-overlapped moving-windows of 512 pixels. Note that tiles with less than 20% of tissue were excluded. The dataset is divided into three class-wise balanced groups for training, validation and testing.

In addition, SICAP-MIL includes instance-level annotations, which allow to test the capability of MIL methods to leverage instance classifications in a weakly-supervised manner. To do so, annotated WSIs are kept into the test subset. Note that instance-level labels are obtained from pixel-level annotations done by expert pathologist. Non-cancerous patches are obtained only from benign WSIs, while cancerous patch-level labels are obtained by majority voting of segmentation masks.

You can download it in the following link: [SICAP-MIL](https://cvblab.synology.me/PublicDatabases/SICAP_MIL.zip)

If you find this dataset useful for your research, please consider citing:

**J. Silva-Rodríguez, A. Schmidt, M.A. Sales, R. Molina and V. Naranjo, "Proportion constrained weakly supervised histopathology image classification", Computers in Biology and Medicine, vol. 147, p. 105714, 2022.** [(paper)](https://www.sciencedirect.com/science/article/pii/S0010482522004930)

To use the methods introduced in this repository, we encourage to move the obtained folder to './data/'. 

## Instance-based MIL Classification

This project aims to levarage precise instance-level labels on gigapixel WSIs using only global labels during training, in the multi-label scenario of prostate cancer. To do so, we base our methods on an instance-based MIL. In particular, max-aggregation of instance-level predictions usually gets more precise results than other aggregation methods such as mean-aggregation. You can train this baseline using the following code:

```
python main.py --experiment_name instance_max --aggregation max --mode instance --scheduler True --early_stopping True --criterion auc --epochs 100
```

## Self-Supervised Learning

Instance-level labels obtained using max-aggregation lack of large sensitivity, since max-aggregation only focus on the most distriminant instances. To alleviate this issue, training an Student model on distilled pseudolabels from the MIL-trained Teacher has shown promising results:

![ssl](https://github.com/jusiro/mil_histology/blob/main/images/student_method.png)

```
python train_student.py --experiment_name instance_max
```

This method is described in detail in the following article:

**J. Silva-Rodríguez, A. Colomer, J. Dolz and V. Naranjo, "Self-Learning for Weakly Supervised Gleason Grading of Local Patterns," in IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 8, pp. 3094-3104, 2021.** [(paper)](https://ieeexplore.ieee.org/abstract/document/9361085)[(arXiv)](https://arxiv.org/abs/2105.10420)

## Proportion Constraints
Weakly supervised learning methods usually require large amounts od data to perform properly.In this project, we propose to introduce proportion priors per WSI as additional, weak information, to train realiable deep learning models. Concretely, we take advantadge of the Gleason scoring system, which defines a primary and secondary class per bag based on tissue proportions. Thus, we use inequality constraints to ensure that the percentage of positive predictions for the primary class is larger than the one for the secondary grade. Our formulation is flexible, and might deal with other problems, where more constraints are known regarding relative class proportions. You can train the proposed model using the following code.

![pc](https://github.com/jusiro/mil_histology/blob/main/images/proportion_method.png)

```
python main.py --experiment_name instance_max_Constrained --aggregation max --mode instance --pMIL True --alpha_ce 1 --alpha_ic 0.1 --alpha_pc 1 --t_ic 15 --t_pc 5 --scheduler True --early_stopping True --criterion z --epochs 100
```

Later, you can also train the previously-presented self-supervised Student model to refine the obtained model. Trained models are available from the following link: [models](https://cvblab.synology.me/PublicDatabases/SICAP_MIL_models.zip).

This method is described in detail in the following article:

**J. Silva-Rodríguez, A. Schmidt, M.A. Sales, R. Molina and V. Naranjo, "Proportion constrained weakly supervised histopathology image classification", Computers in Biology and Medicine, vol. 147, p. 105714, 2022.** [(paper)](https://www.sciencedirect.com/science/article/pii/S0010482522004930)

## Visualizations

Finally, you can produce visualization of instance-level predicitons through the following code:

```
python produce_visualizations.py --experiment_name instance_max_Constrained
```
![visualizations](https://github.com/jusiro/mil_histology/blob/main/images/visualzations.png)

## Contact
For further questions or details, please directly reach out to Julio Silva-Rodríguez
(jusiro95@gmail.com)
