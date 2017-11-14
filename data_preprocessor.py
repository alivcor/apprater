import json
from pprint import pprint

data = json.load(open('dataset_100/1/meta.json'))


price = data["price"]
isFree = data["free"]
version = data["version"]
offersIAP = data["offersIAP"]
adSupported = data["adSupported"]
androidVersion = data["androidVersion"]
contentRating = data["contentRating"]
description = data["description"]
score = data["score"]

print price,
