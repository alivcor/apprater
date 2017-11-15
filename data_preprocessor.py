import json
import Constants
import os
from pprint import pprint
import description_rater as dr

path = os.getcwd()
Train_data_path= os.path.join(path+'/'+'dataset_100'+'/'+Constants.training_data)

f=open(Train_data_path,'w')
f.write('id'+',')
f.write('price'+',')
f.write('isFree'+',')
f.write('version'+',')
f.write('offersIAP'+',')
f.write('adSupported'+',')
f.write('androidVersion'+',')
f.write('contentRating'+',')
f.write('textScore'+',')
f.write('hasVideo'+',')
f.write('score')
f.write('\n')

count=0
while count < Constants.Max_Number_Of_Apps:
	count=count+1
	data = json.load(open('dataset_100/'+str(count)+'/meta.json'))
	Video_file = os.path.join('dataset_100/'+str(count)+'/'+'*.mp4')
	price = data["price"]
	if price[0] == '$':
		price=price[1:] 
	if 'free' in data.keys():
		if data["free"] == True:
			isFree=1
		else:
			isFree=0

	if 'version' in data.keys():
		version = data["version"]
		version = version[0]
		if version == 'V':
			version = 1

	else:
		version=0

	if 'offersIAP' in data.keys():
		if data["offersIAP"] == True:
			offersIAP=1
		else:
			offersIAP=0


	#offersIAP = data["offersIAP"]
	#adSupported = data["adSupported"]
	if 'adSupported' in data.keys():
		if data["adSupported"] == True:
			adSupported=1
		else:
			adSupported=0

	androidVersion = data["androidVersion"]
	if len(androidVersion) > 3:
		androidVersion=androidVersion[:2]
		if androidVersion == 'VA':
			androidVersion=1


	contentRating = data["contentRating"]
	if contentRating == 'Mature 17+':
		contentRating = 2
	elif (contentRating == 'Teen'):
		contentRating = 3
	elif (contentRating == 'Unrated'):
		contentRating = 0
	elif (contentRating == 'Everyone'):
		contentRating = 4
	elif (contentRating == 'Everyone 10+'):
		contentRating = 1



	if(os.path.isfile(Video_file)):

		hasVideo=1
	else:
		#print Video_file
		hasVideo=0
	description = data["description"]
	print description
	description_score = dr.rate_description(description)
	score = data["score"]

	f.write(str(count)+',')
	f.write(price+',')
	f.write(str(isFree)+',')
	f.write(str(version)+',')
	f.write(str(offersIAP)+',')
	f.write(str(adSupported)+',')
	f.write(str(androidVersion)+',')
	f.write(str(contentRating)+',')
	f.write(str(description_score)+',')
	f.write(str(hasVideo)+',')
	f.write(str(score))
	f.write('\n')