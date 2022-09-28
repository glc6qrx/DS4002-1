# Predicting the Age of NBA Athletes Using a Convolutional Neural Network

The purpose of this repository is to assist readers in reproducing our results on age classification for facial images of the NBA population. The repository consists of:

## SRC
### Code Building

**Test Data Creation** 
To obtain the pictures and ages of all current NBA players, we wrote a program named 'grabAndName.py' to automate the process. All NBA players' most recent picturee is available here: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/PLAYERID.png (where you replace PLAYERID with the Player's ID). All the Player IDs can be found [here](http://data.nba.net/data/10s/prod/v1/2019/players.json). We tidied the JSON dataset in Excel to remove duplicates and players who were not currently active. We also used Excel’s functionality to get the age of each player from their Date of Birth along with the instance of that age (which will be significant later when we are analyzing the accuracy of the algorithm for different age groups). Then we wrote grabAndName.py to download all the most recent pictures for all active players and name the files in the format: age_ageInstance.

**Model Training** 
The code to train our Deep Learning Age Detection Model was created by [DigitalSreeni](https://www.youtube.com/watch?v=rdjWDAYt98s). The code to test and evaluate the model was created by Catherine Schuster. All model code is consolidated in AgeDetection.py

### Code Usage
If you find our models or code useful, please add suitable reference to our project and in your work.

## Data 
### Training Data: UTK FACE
The data used to train the age classifcation model comes from the [UTK Face](https://susanqq.github.io/UTKFace/) database. The specific data used is the "Aligned & Cropped Faces" file. 

The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
| Name        | Description                                                                                                         |
| ------------|---------------------------------------------------------------------------------------------------------------------|
| age         | is an integer from 0 to 116, indicating the age                                                                     | 
| gender      | is either 0 (male) or 1 (female)                                                                                    | 
| race        | is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern) |
| date&time   | is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace                  |


### Test Data: NBA 
The data used to test the age classification model was built using NBA player data [here](https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/PLAYERID.png) and [here](http://data.nba.net/data/10s/prod/v1/2019/players.json). Folder containing most recent pictures of all active NBA players. All files are named in the format “age_instance.”
| Name        | Description                                                                                                         |
| ------------|---------------------------------------------------------------------------------------------------------------------|
| image       | facial image of NBA player                                                                                          | 
| age         | age of the player.                                                                                                  | 

## FIGURES 
Scatterplots of (1) training accuracy and (2) training validation loss during model training.

## References 
DigitalSreeni, “240 - Deep Learning training for age and gender detection.,” YouTube.com, Oct. 20, 2021. [Online]. Available: https://www.youtube.com/watch?v=rdjWDAYt98s. [Accessed Sept. 12, 2022].

Y. Song, “UTKFace | Large Scale Face Dataset,” [Online]. Available: https://susanqq.github.io/UTKFace/. [Accessed Sept. 12, 2022].

