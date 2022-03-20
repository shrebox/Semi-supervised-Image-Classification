# Task 3: Attention loss

Files: main.py, utils.py, LR_scheduler.py, dataloader.py, randomaugment.py, model/wrn.py
Saved models: saved_models/{_dataset-labelled-samples_} (Present in the main directory)

* In the main.py, we have all the parameters that are read from the command-line arguments which in turn calls the Trainer() class in utils.py which contains all the function for training and testing. 
* LR_scheduler.py contains the code for learning rate scheduling.
* randomaugment.py contains the code for random augmentation for Fixmatch.

## Important parameters:

* --dataset
To specify the dataset to use (cifar-10 or cifar100).

* --train-batch
Determine the size of the train batch. Also, the size of train and test batch will be same set by this parameter.

* --att-loss
This flag is used to enable the attention loss.

* --fixed-lr
This flag is used to disable the learning rate scheduling, and use a fixed learning rate.

## Example commands:

python main.py --dataset cifar10 --train-batch 250 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --attention-heads 2 --att-loss

python main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --attention-heads 2 --att-loss

python main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --fixed-lr --attention-heads 4 --att-loss

