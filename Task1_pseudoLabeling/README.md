# Task 1: Pseudo-labelling

Files: main.py, utils.py, dataloader.py, model/wrn.py
Saved models: saved_models/{_dataset-labelled-samples_} (Present in the main directory)

* In the main.py, we have all the parameters that are read from the command-line arguments which in turn calls the Trainer() class in utils.py which contains all the function for training and testing. 
## Important parameters:

* --dataset
To specify the dataset to use (cifar-10 or cifar100).

* --train-batch
Determine the size of the train batch. Also, the size of train and test batch will be same set by this parameter.

* --lr
The learning rate for the model.

* --epochs
The total number of epochs to train the model. if warmup is used, the number of epochs in which the model will train on pseudo labels will be epcohs-warmup.

* --warmup-epochs
The number of epochs in which the model is trained only on the labeled data before starting the ssl training.


## Example commands:

python ./main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 2500  --threshold 0.95 --lr 0.0005

python ./main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 10000  --threshold 0.75 --lr 0.0005


