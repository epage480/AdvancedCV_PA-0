Part 1

Train_Model.py can be used to test the performance of 3 models on the SVHN dataset.
The program should automatically download the dataset if it is not already in the same directory.
The following 3 lines can be run within the directory to replicate the 1st series of experiments.

python3 "Train_Model.py" --model Model1 --save_path "Model_Saves/Model1/Save" --logs_path "tensorflow_logs/Model1/
python3 "Train_Model.py" --model Model2 --save_path "Model_Saves/Model2/Save" --logs_path "tensorflow_logs/Model2/
python3 "Train_Model.py" --model Model3 --save_path "Model_Saves/Model3/Save" --logs_path "tensorflow_logs/Model3/
-----------------------------------------------------------------------------------------------------------------------

Part 2
Train_Model.py is also used for part 2, the data_percent argument can be used to vary the amount of training data used.
The following 3 lines can be run within the same directory as the 1st to replicate the 2nd set of experiments.

python3 "Train_Model.py" --model Model3 --save_path "Model_Saves/ReducedData1/Data10_" --data_percent 1
python3 "Train_Model.py" --model Model3 --save_path "Model_Saves/ReducedData2/Data7_" --data_percent .7
python3 "Train_Model.py" --model Model3 --save_path "Model_Saves/ReducedData3/Data3_" --data_percent .3

Complete details of Train_Model.py shown below

usage: Train_Model.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                      [--save_path SAVE_PATH] [--model MODEL]
                      [--logs_path LOGS_PATH] [--data_percent DATA_PERCENT]
optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size, number of samples used per training
                        iteration
  --epochs EPOCHS       Epochs, number of full passes through data set
  --save_path SAVE_PATH
                        Model save path
  --model MODEL         Select the model architecture: Model1, Model2, or
                        Model3 (smallest to largest)
  --logs_path LOGS_PATH
                        Tensorboard log save path
  --data_percent DATA_PERCENT
                        Percent of the data used to train the network

-----------------------------------------------------------------------------------------------------------------------

Part 3

Tune_AlexNet.py is used for part 3 and can be run without arguments.

Complete details of Tune_AlexNet.py shown below

usage: Tune_AlexNet.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                       [--save_path SAVE_PATH] [--logs_path LOGS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size, number of samples used per training
                        iteration
  --epochs EPOCHS       Epochs, number of full passes through data set
  --save_path SAVE_PATH
                        Model save path
  --logs_path LOGS_PATH
                        Tensorboard log save path

