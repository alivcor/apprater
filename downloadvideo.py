import json
from pprint import pprint
import youtube_dl
import Constants
import os

path=getcwd()
App_Id_File_Path = os.path.join(path+'/'+Constants.App_Id_File_Path)
count =0

with open(App_Id_File_Path) as f:
	Appid = f.readlines()
Appid = [x.strip() for x in Appid]

for app in Appid:
	count = count+1
	node_command = "node " + app + " "+ str(count) 
	os.system(node_command)
	data = json.loads(open(str(count)+".json","r").read())
	video_url = data['video']
	print data['video']
	print "Downloading!"
	ydl_opts = {}
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    	ydl.download([video_url])