# Virtual Adversarial Training (VAT)

Files: main.py, vat.py, utils.py, LR_scheduler.py, dataloader.py, model/wrn.py
Saved models: saved_models/{_dataset-labelled-samples_} (Present in the main directory)
Additional files: Run outputs: server_run_outputs/{.err, .out, .log} from the server for the results mentioned in the report.

* In the main.py, we have all the parameters that are read from the command-line arguments which in turn calls the Trainer() class in utils.py which contains all the function for training and testing. 
* VAT loss is calulated in the vat.py file.
* LR_scheduler.py contains the code for learning rate scheduling.

## Important parameters:

* --dataset
To specify the dataset to use (default: cifar10).

* --train-batch
Determine the size of the train batch. Also, the size of train and test batch will be same set by this parameter.

Also, 
> for labeled data with size 250 set the batch size to 250
> for labeled data with more than 250 set the batch size to 512
> set warmup-epochs to 35 and epochs to 300

* --warmup-epochs
Number of epochs the model will be trained first on the labelled samples only. Note, the warmup epochs are counted in the total number of epochs.

* --epochs
The total number of epochs for which the model has to be trained.

* --num-labeled
This parameter is to specify the number of labelled samples to be provided to the model while training.

* --use_entmin 
This boolean (True/False) flag parameter is used to toggle the addition of conditional entropy minimization term in the vat loss.

* --vat_xi
Specify the perturbation budget for the vat loss.

* --vat_eps
Specify the final perturbation budget for the vat loss.

* --vat_iter
Specify the power iteration (K) for the vat loss.

## Example commands:

Note: You need to be in the Task 2 (VAT) directory.

python main.py --train-batch 1024 --test-batch 1024 --warmup-epochs 35 --epochs 300 --num-labeled 4000 --use_entmin False --vat_xi 10.0 --vat_eps 10.0 --vat_iter 1

python ./Task2_VAT/main.py --train-batch 250 --test-batch 1024 --warmup-epochs 40 --epochs 50 --num-labeled 250 

python ./Task2_VAT/main.py --dataset cifar100 --train-batch 1024 --test-batch 1024 --warmup-epochs 35 --epochs 300 --num-labeled 2500 

python ./Task2_VAT/main.py --dataset cifar100 --train-batch 1024 --test-batch 1024 --warmup-epochs 35 --epochs 300 --num-labeled 10000

## Additional notes

* For running on the university server, we have also provided experiments.sub and start.sh which are required to run multiple experiments. Also, '--log-dir' (default: ./runs) and '--model-dir' (default: ./saved_models) needs to be specificed which has to be same in the start.sh.
* start.sh can be used to pass multiple commands at once. For example, all the above example commands can be passed together and the results will be zipped in the runs.tar and saved_model.tar for each command in a single job.
* We have shipped the saved models for {dataset, labelled_samples, epochs}: {CIFAR-10, 4000, 300}, {CIFAR-10, 250, 50}, {CIFAR-10, 10000, 100}, {CIFAR-10, 2500, 300}. 
* For two out of four saved models, the models are trained for lesser number of epochs as the initial saved models were over-written but the logs out the output contain the run results. 
* The lesser number of epochs doesn't hurt the performance as we saw results saturating around 120 epochs.
