import json
from pprint import pprint
import youtube_dl


data = json.loads(open("foo.json","r").read())
video_url = data['video']
print data['video']
print "Downloading!"
ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])