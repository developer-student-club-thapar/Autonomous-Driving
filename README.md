<div align = "center">

<img height=200px src= "https://github.com/developer-student-club-thapar/officialWebsite/blob/master/Frontend/src/assets/dsc_logo.png">

<h1>DEVELOPER STUDENT CLUBS TIET</h1>

<a href="https://medium.com/developer-student-clubs-tiet"><img src="https://github.com/aritraroy/social-icons/blob/master/medium-icon.png?raw=true" width="60"></a>
<a href="https://twitter.com/dsctiet"><img src="https://github.com/aritraroy/social-icons/blob/master/twitter-icon.png?raw=true" width="60"></a>
<a href="https://www.linkedin.com/company/developer-student-club-thapar"><img src="https://github.com/aritraroy/social-icons/blob/master/linkedin-icon.png?raw=true" width="60"></a>
<a href="https://facebook.com/dscthapar"><img src="https://github.com/aritraroy/social-icons/blob/master/facebook-icon.png?raw=true" width="60"></a>
<a href="https://instagram.com/dsc.tiet"><img src="https://github.com/aritraroy/social-icons/blob/master/instagram-icon.png?raw=true" width="60"></a>

</div>


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
[Link for Demo 1](../Demo 1.gif)<br />
[Link for Demo 2](../demo/Demo 2.gif)<br />
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
