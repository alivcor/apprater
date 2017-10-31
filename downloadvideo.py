import json
from pprint import pprint
import youtube_dl
import Constants
import os

path=os.getcwd()
App_Id_File_Path = os.path.join(path+'/'+Constants.App_Id_File_Path)
#Video_File = os.path.join(path+'/'+Constants.Videos)
count =0

with open(App_Id_File_Path) as f:
	Appid = f.readlines()
Appid = [x.strip() for x in Appid]

for app in Appid:
	count = count+1
	node_command = "node scraper.js " + app + " "+ str(count) 
	os.system(node_command)
	Video_File = os.path.join(path+'/'+Constants.Videos_File_Path+'/'+str(count)+'/'+"meta.json")
	data = json.loads(open(Video_File,"r").read())
 	video_url = data['video']
 	print data['video']
 	print "Downloading!"
 	dow_dir=path+'/'+Constants.Videos_File_Path+'/'+str(count)
 	ydl_opts = {"o": dow_dir}
 	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
 		ydl.download([video_url])
 	os.system("mv *.mp4 "+ dow_dir)