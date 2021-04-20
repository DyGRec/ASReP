# ASReP
This is our TensorFlow implementation for the paper:
SIGIR'21 ([PDF](coming soon)
```bibtex
@inproceedings{liu2021augmenting,
  title={Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer},
  author={Liu, Zhiwei* and Fan, Ziwei* and Wang, Yu and Yu, Philip S.},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

The code is implemented based on Tensorflow version of [SASRec](https://github.com/kang205/SASRec).
Please cite our paper if you use the code.

## Environment Setup
The code is tested under a Linux desktop (w/ GTX 1080 Ti GPU) with TensorFlow 1.12 and Python 3.6.
Create the requirement with the requirements.txt

## Datasets
We use the Amazon Review datasets Beauty and Cell_Phones_and_Accessories. The data split is done in the
leave-one-out setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).

### Data Preprocessing
Use the DataProcessing.py under the data/, and make sure you change the DATASET variable
value to your dataset name, then you run:
```
python DataProcessing.py
```
You will find the processed dataset in the directory with the name of your input dataset.


## Baby Dataset Pre-training and Prediction
### Reversely Pre-training and Short Sequence Augmentation
Pre-train the model and output 20 items for sequences with length <= 20.
```
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20
```
### Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
```
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 15 --M 18
```

## Cell_Phones_and_Accessories Dataset Pre-training and Prediction
### Reversed Pre-train and Short Sequence Augmentation
Pre-train the model and output 20 items for sequences with length <= 20.
```
python main.py --dataset=Cell_Phones_and_Accessories --train_dir=default --lr=0.001 --hidden_units=32 --maxlen=100 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20
```
### Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
```
python main.py --dataset=Cell_Phones_and_Accessories --train_dir=default --lr=0.001 --hidden_units=32 --maxlen=100 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 100 --reversed_pretrain 1  --aug_traindata 17 --M 18
```

