# Predicting the Age of NBA Athletes Using a Convolutional Neural Network

The purpose of this repository is to assist readers in reproducing our results on age classification for facial images of the NBA population. The repository consists of:

## SRC
### Code Building


### Code Usage
If you find our models or code useful, please add suitable reference to our project in your work.

Make an H3 section for Installing/Building your code
Make an H3 section for Usage of your code · 

## Data 
### Training Data: UTK FACE
The data used to train the age classifcation model comes from the [UTK Face](https://susanqq.github.io/UTKFace/) database. The specific data used is the "Aligned & Cropped Faces" file. 

The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
| :---        |    :----:   | 
| age         | is an integer from 0 to 116, indicating the age | 
| gender      | is either 0 (male) or 1 (female)       | 
| race        | is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern) |
| date&time   | is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace        |


### Test Data: NBA 
To obtain the pictures and ages of all current NBA players, we wrote a program named 'grabAndName.py' to automate the process. All NBA players' most recent picture is available at the URL: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/PLAYERID.png. All the Player IDs can be found at http://data.nba.net/data/10s/prod/v1/2019/players.json. We tidied the JSON dataset in Excel to remove duplicates and players who were not currently active. We also used Excel’s functionality to get the age of each player from their Date of Birth along with the instance of that age (which will be significant later when we are analyzing the accuracy of the algorithm for different age groups). Then we wrote grabAndName.py to download all the most recent pictures for all active players and name the files in the format: age_ageInstance.

## FIGURES 
This will be in progress when MI3 is complete and finished during MI4 
Table of contents describing all figures produced and summarizing their takeaways
Use markdown table formatting · 

## References 
All references should be listed at the end of the Readme.md file (Use IEEE Documentation style (link)) 
Include any acknowledgements 
Include (by link) your MI1 and MI2 assignments

