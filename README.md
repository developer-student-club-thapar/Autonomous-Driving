# Autonomous-Driving
This repository contains the code for steering a self-driving vehicle.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installation and Setup

* Fork the repo and clone it.
```
git clone https://github.com/developer-student-club-thapar/Autonomous-Driving.git
```
* Go in the repo and setup virtual environment using `python -m virtualenv env` or by using anaconda <br />`conda create --name env`  
* Then activate the environment using `source env/Scripts/activate` (For Windows Users using CMD or Powershell `env/Scripts/activate`) or
`conda activate env`
* Then install the necessary required packages from requirements.txt
```
pip install -r requirements.txt
```
### Running the demo
![Demo](demo/demo.gif)<br />
Run the following command : 
```
python autolabeler.py --basedir "all" --datafile "angles.csv" --savefile "./output.csv"
```
* basedir : Path to folder that dataset folder
* datafile : Path to angles.csv
* savefile : Path to store output.csv

### Training the Neural Network
* Make sure your dataset-images are in folder named 'all' at the root directory.
* Make sure your angles.csv and actions.csv are in the root directory named 'angles.csv' and 'actions.csv' respectively.
* The models.py contain architectures of 4-different neural networks named: 
   * comma.ai
   * pretrained_vgg16
   * nvidia-dave2
   * 3D-CNN
 * Run the followuing command : 
 ```
 python steering-model.py --model "model_name from above list"
 ```
 For Example : 
 ```
 python steering-model.py --model "comma.ai"
 ```
 ***NOTE : Use exact name as mentioned above to avoid unnecessary bugs.***
## LICENSE
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



