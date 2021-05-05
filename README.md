#  Deep Learning 
A repository of Deep Learning Projects and Tasks.

Feel free to use the codes and shoot me a mail in case there are issues.

"Gaussian NB" and Logistic Regression are basic introduction to ML tasks 

Video classification, "Resnet" and "Y-net" are different image classification projects on CIFAR10 and CIFAR100. 

"RNN_LSTM for Image Captioning" is a code base that can be used to experiment with various configurations of RNN, LSTM and GRU with various CNNs for the task.  
These include implementations(from scratch) of networks using tenserflow and keras for the task and experiments of "Transfer learning"  from pretrained netwroks 

## Fashion MNIST classifier 

Platform and Packages used are as listed below.
Python3 is a pre-requisite. The following packages will be required as well:
tensoflow 2 and keras is used to build and train the network
1. tf2
2. sklearn
3. matplotlib

This script can be run from the command line as shown below

```shell
python3 fashion_mnist_classifier.py -e "epochs" -bs "batch_size" -tmo -mp "path to saved model"

```

example for running with train:
 
```shell
python3 fashion_mnist_classifier.py -e 200 -bs 1024 

```
example for only test:
 
```shell
python3 fashion_mnist_classifier.py -tmo -mp "..\..\saved_models"

```

Handle -e / --epochs is the number of training epochs.
Handle -bs / --batch_size is the batch_size for training.
Handle -tmo / --test_model_only is a flg to run on tests and report metrics without training, this can be doen when we have a saved model.
Handle -mp / --model_path is the path to saved model.
All arguments are optional.
The default run will without any args will train 


