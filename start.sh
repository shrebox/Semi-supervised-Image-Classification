nvidia-smi

## Task 1

## CIFAR-10
# python ./Task1_pseudoLabeling/main.py --train-batch 250 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 250  --threshold 0.95 --lr 0.001 --min-lr 0.00000005
# python ./Task1_pseudoLabeling/main.py --train-batch 250 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 250  --threshold 0.75 --lr 0.001 --min-lr 0.00000005
# python ./Task1_pseudoLabeling/main.py --train-batch 250 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 250  --threshold 0.6 --lr 0.001 --min-lr 0.00000005

## CIFAR-100
# python ./Task1_pseudoLabeling/main.py --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 4000  --threshold 0.95 --lr 0.001 --min-lr 0.00000005
# python ./Task1_pseudoLabeling/main.py --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 4000  --threshold 0.75 --lr 0.001 --min-lr 0.00000005
# python ./Task1_pseudoLabeling/main.py --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 4000  --threshold 0.6 --lr 0.001 --min-lr 0.00000005

## Task 2

## CIFAR-10
python ./Task2_VAT/main.py --train-batch 1024 --test-batch 1024 --warmup-epochs 50 --epochs 300 --num-labeled 4000 --use_entmin False --vat_xi 10.0 --vat_eps 10.0 --vat_iter 1
python ./Task2_VAT/main.py --train-batch 250 --test-batch 1024 --warmup-epochs 40 --epochs 50 --num-labeled 250 

## CIFAR-100
python ./Task2_VAT/main.py --dataset cifar100 --train-batch 1024 --test-batch 1024 --warmup-epochs 35 --epochs 300 --num-labeled 2500 
python ./Task2_VAT/main.py --dataset cifar100 --train-batch 1024 --test-batch 1024 --warmup-epochs 35 --epochs 300 --num-labeled 10000

## Task 3

python main.py --dataset cifar10 --train-batch 250 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --attention-heads 2 --att-loss
python main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --attention-heads 2 --att-loss
python main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --num-workers 2 --epochs 100 --num-labeled 250  --threshold 0.6 --lr 0.003 --fixed-lr --attention-heads 4 --att-loss

## For saving the tensorboard files and models 
## Note keep the names runs and saved_models same as the parameter --log-dir and --model-dir (if want to store with some other name/location)
tar zcf runs.tar.gz runs
tar zcf saved_models.tar.gz saved_models
mv runs.tar.gz ../
mv saved.tar.gz ../
