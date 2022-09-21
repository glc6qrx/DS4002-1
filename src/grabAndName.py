import urllib.request
import os
import pandas as pd
import shutil

import requests as requests

players = pd.read_csv('players.csv')

personIdList = players.personId.values.tolist()
cleanedPersonIdList = [x for x in personIdList if x == x]
cleanedPersonIdListInt = list(map(int, cleanedPersonIdList))

personAgeList = players.age.values.tolist()
cleanedPersonAgeList = [x for x in personAgeList if x == x]
cleanedPersonAgeListInt = list(map(int, cleanedPersonAgeList))

personAgeInstanceList = players.age_instance.values.tolist()
cleanedPersonAgeInstanceList = [x for x in personAgeInstanceList if x == x]
cleanedPersonAgeInstanceListInt = list(map(int, cleanedPersonAgeInstanceList))

url = "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/"
for index, personId in enumerate(cleanedPersonIdListInt):
    url_source = url + str(personId) + ".png"
    img_data = requests.get(url_source).content
    with open(str(cleanedPersonAgeListInt[index]) + "_" + str(cleanedPersonAgeInstanceListInt[index]) + ".png", 'wb') as handler:
        handler.write(img_data)



