# Project 1 MTH 5320 
## David Nieves-Acaron
## 13 October 2021


## Problem description in report (What are you trying to do?  What is your data? What shape is it?)

## Introduction

This project was in part inspired by a Veritasium video detailing how a video can go viral or in a more general sense, achieve a certain number of views. The problem, in a neural networks point of view, would be giving the network certain inputs, and having it predict the number of views or a function of the number of views. Therefore, in a primtive sense, I am trying to predict some sense of how many views a video will get using a feed forward neural network. The data is a set of features from a set of YouTube videos. The shape of the total data is (Videos x Features) where about 63,938 videos were collected, and 12 features were collected, thus making it of a (63,938 x 12) shape. 

However, it must be kept in mind that this is not how the input data is handled when entering it into the neural network. The specific shapes used for inputs will be discussed when showcasing the classification code.


My hypothesis on what makes a video viral is that there are four or so main components to it:

1. Thumbnail 
2. Title
3. Channel
4. YouTube Algorithm Magic
5. The actual content of the video

Given the limitations imposed by this project, I had to make do with some aspects of 3 and 5. Part of the reason why I chose to focus on data collection for this project is because I would like to build on top of the effort realized in this project to be able to perform inference on other parts of the video, such as the Thumbnail, and the Title, as we learn about other types of neural networks such as CNNs. 


## Data Preparation

The data consisted of the following features 

```
{
          "id" : str,
          "game_year" : str,
          "game_name" : str,
          "likes" : int,
          "views" : int,
          "dislikes" : int,
          "l/d" : double,
          "v/s" : double,
          "l/s" : double,
          "uploadDate" : ISO 8601 timestamp,
          "channel" : str,
          "uploader" : str
}


```

for a selection of about 10,000 videos (or less if the channel had less than that or if something impeded the data collection) from each channel. The collection process made use of the YouTube Data API as well as of the ELK stack (ElasticSearch, LogStash, and Kibana)in order to be able to effectively search through the process. The ELK stack was hosted in AWS (Amazon Web Services) in order to simplify the setup process. 

Data Collection took three parts:
- Gathering a list of "gaming" channels on YouTube via their ID's
- Getting a selected list of video ID's for those channels and writing that list to a local file
- Iterating through the aforementioned list, gathering the necessary video information via a web scraper, and uploading the data to ElasticSearch.

The parts will now be explained in more detail.

## First Part
I decided to gather some youtube channels big and small in order to represent the different aspects of the gaming community. I specifically chose to focus more on individuals who run channels and who primarily showcase themselves playing games. This is opposed to something like IGN, where although it features gaming, I feel that views come primarily from people wanting news about new releases rather than anything else. 


Among some of the Youtube Gaming Channel "Types" that I tried to emphasize when collecting channel ID's were:

- The big personality type (ie. PewDiePie, Markiplier, JackSepticEye, SSundae, Auron etc...)
- The "dedicated to one game" type
    * many of them exclusively Minecraft, but competitive players also fall into this category such as sponsored CS Go, and League of Legends players, etc... who have YouTube channels and receive an audience but fail to break out of a niche.
- The variety gamer who has a history but has not broken out of his primary audience 
- International channels such as those from Russia, Japan, Germany, Spain, etc...
- Let's Players such as TheRadBrad, MKIceAndFire, ZachScottGames, etc...

## Second Part
Gathering a list of video ID's required the use of the Youtube Data API, which has a certain quota that its developers need to satisfy. Specifically, one is allotted 10,000 units per day, where different operations cost different units. The latter is a graph showcasing the different costs associated with each operation. ([Youtube Data API Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost))


|Quota costs            |             ||
|-----------------------|-------------------|------|
|resource               |method             |cost  |
|activities             |list               |1     |
|captions               |list               |50    |
|                       |insert             |400   |
|                       |update             |450   |
|                       |delete             |50    |
|channelBanners         |insert             |50    |
|channels               |list               |1     |
|                       |update             |50    |
|channelSections        |list               |1     |
|                       |insert             |50    |
|                       |update             |50    |
|                       |delete             |50    |
|comments               |list               |1     |
|                       |insert             |50    |
|                       |update             |50    |
|                       |markAsSpam         |50    |
|                       |setModerationStatus|50    |
|                       |delete             |50    |
|commentThreads         |list               |1     |
|                       |insert             |50    |
|                       |update             |50    |
|guideCategories        |list               |1     |
|i18nLanguages          |list               |1     |
|i18nRegions            |list               |1     |
|members                |list               |1     |
|membershipsLevels      |list               |1     |
|playlistItems          |list               |1     |
|                       |insert             |50    |
|                       |update             |50    |
|                       |delete             |50    |
|playlists              |list               |1     |
|                       |insert             |50    |
|                       |update             |50    |
|                       |delete             |50    |
|search                 |list               |100   |
|subscriptions          |list               |1     |
|                       |insert             |50    |
|                       |delete             |50    |
|thumbnails             |set                |50    |
|videoAbuseReportReasons|list               |1     |
|videoCategories        |list               |1     |
|videos                 |list               |1     |
|                       |insert             |1600  |
|                       |update             |50    |
|                       |rate               |50    |
|                       |getRating          |1     |
|                       |reportAbuse        |50    |
|                       |delete             |50    |
|watermarks             |set                |50    |
|                       |unset              |50    |



One key detail which was painfully found out while trying to collect data was that searching (using the /search?channel domain and then matching the channel ID) is very expensive (100 points out of 10,000 points) and I would run into the problem of running out of units. However, how would one be able to list the videos from a channel if not by searching for them? The answer is through playlists, for which listing would be much cheaper. Additionally, one strange trick I found was that a default playlist of user uploads (obtained by replacing the second character of the user's channel ID with 'U') could be used to cheaply list all the uploads of a channel as opposed to relying on whatever playlists the owner channel has made (usually none or a much more limited scope than the entire set of uploads).

The following contains the code used to collect the video ID's from the selected list of channel ID's. Please note that it is not perfect, as some video ID's failed, but that is why I added a try catch statement. Additionally, there is no real fear of duplicates since later on when processing the data in order to put it into the NN, the video ID's get put into a set. 

## DataCollectionWorking.py


```python
import os
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle

from requests import api
from apiclient.discovery import build

from tqdm import tqdm
import requests
import json
import re
import time
Api_key = open("apikey.txt", 'r')
Api_key = str(Api_key.read())

# This man saved my life:
# https://www.youtube.com/watch?v=th5_9woFJmk
# Bless him
# Corey Schafer

if os.path.exists('token.pickle'):
    print('Loading Credentials From File...')
    with open('token.pickle', 'rb') as token:
        credentials = pickle.load(token)
        print(credentials.to_json())
else:
    credentials = ""

# Google's Request
from google.auth.transport.requests import Request
# MAKE SURE REDIRECT URI IN GOOGLE CLOUD SERVICE CONSOLE IS http://localhost:8080/ WITH THE SLASH AFTERWARDS

# If there are no valid credentials available, then either refresh the token or log in.
if not credentials or not credentials.valid:
    if credentials and credentials.expired and credentials.refresh_token:
        print('Refreshing Access Token...')
        credentials.refresh(Request())
    else:
        print('Fetching New Tokens...')
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret_3.json',
            scopes=[
                'https://www.googleapis.com/auth/youtube.readonly'
            ]
        )

        flow.run_local_server(port=8080, prompt='consent',
                              authorization_prompt_message='')
        credentials = flow.credentials

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as f:
            print('Saving Credentials for Future Use...')
            print(credentials.to_json())
            pickle.dump(credentials.to_json(), f)






class channelDownloader:
    def __init__(self, *,api_key, channelID, targetAmount):
        self.api_key = api_key
        self.channelID = channelID
        self.targetAmount = targetAmount
        self.counter = 0
        self.total_requests = 0
        url = "https://www.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics%2CbrandingSettings&id=" +  str(self.channelID) + "&key=" + str(self.api_key)
        resp = requests.get(url)
        data = resp.content
        data = json.loads(data)
        # Wikndows Version
        # w = open(".\\channels.txt","a")
        w = open("./channels.txt","a")
        w.write( str(self.channelID) + "\n")
        w.close()
        # Windows Version
        # os.mkdir(".\\channels\\" + str(self.channelID))
        os.mkdir("./channels/" + str(self.channelID))
        

        # Get subscriber Count, videoCount
        # Not really necessary just yet
        self.subscriberCount = int(data["items"][0]["statistics"]["subscriberCount"])
        # self.moderateComments = int(data["items"][0]["brandingSettings"]["moderateComments"])
        # self.country = int(data["items"][0]["country"])
        self.videoCount = int(data["items"][0]["statistics"]["videoCount"])
        # print("Number of videos is ", self.videoCount)
        self.nextPageToken = ""


    def iterateOverChannelPlaylists(self):
        # print("CHANNEL: ", self.channelID, " #######################################")
        videos = []
        videos_request = youtube.playlistItems().list(part="status,id,snippet,contentDetails", playlistId=self.channelID.replace(self.channelID[1], "U"), maxResults=50)
        response_videos = videos_request.execute()
        # Windows Version
        # w = open(".\\channels\\" + str(self.channelID) + "\\videos.txt","a")
        w = open("./channels/" + str(self.channelID) + "/videos.txt","a")
        for i in tqdm(range(self.videoCount//50 + 1)):
            if(i != 0):
                # In here is where the Uploads playlist of the channel is found
                # playlistId=self.channelID.replace(self.channelID[1], "U")
                # It makes some sence given that most channel ID's look like this:
                # UCyMy3i-BaVOmOwTZskm52Ew
                # Where the UC probably stands for User Content, and by replacing C with U, 
                # it turns into User Uploads
                videos_request = youtube.playlistItems().list(part="status,id,snippet,contentDetails", playlistId=self.channelID.replace(self.channelID[1], "U"),pageToken=self.nextPageToken, maxResults=50)
                response_videos = videos_request.execute()
                # Used for iterating over page tokens (50 items per page at max)
                self.nextPageToken=response_videos["nextPageToken"]
            for video_count, video in enumerate(response_videos["items"]):
                # Append video ID into file under /channels/channelID/videos.txt
                w.write(str(video["contentDetails"]["videoId"]) + "\n")
                # If the channel has less than the target amount or if the number of collected videos surpasses
                # the target amount, stop.
                if((i)*50 + video_count >= self.targetAmount - 1 or (i)*50 + video_count >= self.videoCount):
                    w.close()
                    return videos
        w.close()
            
            
        
                



youtube = build("youtube", "v3", credentials = credentials)



# Here are the different lists I collected. Many of these channels fail out when collecting the uploads
#  I don't think the UU trick mentioned above works for all channels, is why. The final list was actually much smaller
# at about 58 channels, and not all of the videos belonging to those channels were collected.

# New list. I added some videos from other games such as smash bros, etc... 
# now at 112250 videos. 
channels = [
    "UCam8T03EOFBsNdR0thrFHdQ",
    "UC-cnbLlplnXA4oc7rR21qzg",
    "UCL3r1JcBQM1cmyMhMFm6XkQ",
    "UCcZUc0Wbt4EPVktbB8FrugQ",
    "UCoIXnB865l9Ex9zs4OIXTdQ",
    "UCvAR_BDFclXJ1Q9ndUhWLCA",
    "UCIIPl-DSCC0prKxGGnJrdGQ",
    "UCuJyaxv7V-HK4_qQzNK_BXQ",
    "UCRD2CerUvgKHQ-wXWXGPJ-w",
    "UCCO9_1dqBcBOXa5nB83--pg",
    "UCj1J3QuIftjOq9iv_rr7Egw",
    "UCVFWJkN7L45x8gZTMXu2UWw",
    "UCWzLmNWhgeh3h1j-M-Isy0g",
    "UCfgh3Ul_dG6plQ7rzuOLx-w"
]


# new list added spanish, japanese, german, australian, french youtubers
# now at 83621 videos
"""
"UCXGPGV90SPduyn9LVX9s7Kw",
"UCcG-OmRBrHqiwI91ZbhZt2w",
"UC7tHXQXWImq_0I3PCQ8Udaw",
"UCw1SQ6QRRtfAhrN_cjkrOgA",
"UCWZmCMB7mmKWcXJSIPRhzZw",
"UCYVinkwSX7szARULgYpvhLw",
"UCS5Oz6CHmeoF7vSad0qqXfw",
"UCyMy3i-BaVOmOwTZskm52Ew",
"UC9PD3EIAA-vtGLZgYXG3q0A",
"UCEPuItFWOOJ2o5hTu65NlEg",
"UCdKuE7a2QZeHPhDntXVZ91w",
"UC3C3YOGFjn7Pq3lOCeUFHfg",
"UCUdF4kyAKLyT1xYTjddW5_w",
"UCFR2oaNj02WnXkOgLH0iqOA",
"UCPYJR2EIu0_MJaDeSGwkIVw",
"UCkxctb0jr8vwa4Do6c6su0Q",
"UCpqXJOEqGS-TCnazcHCo0rA",
"UCjlWEMQ2jbTy_2nWq8sEliw",
"UCK0_slr6cUzYFL_dsiuW7Xw",
"UCPbGiUt4Yu8EsaFM-KAmgjg",
"UCKQEfXR_uflA4vDtJHmEIIQ",
"UC5v2QgY2D5tlu8uws23MG4Q",
"UCqg3BHb31w2h77vFuTFmp2w",
"UCH-_hzb2ILSCo9ftVSnrCIQ"
"""

# 50 channels lis, contains big youtubers such as PewDiePie, Markiplier, I think JackSepticEye, SSundae, some LoL dudes, etc...
# GOT ME ABOUT 49422 videos by doing
# for i in tqdm(channels):
#     try:
#         downloader = channelDownloader(api_key=Api_key,channelID=i, targetAmount=10000)
#         downloader.iterateOverChannelPlaylists()
#     except Exception as e:
#         print(e)


"""
[
"UC_wB4WC7FTdXcjPAqVlS7mA"
"UCSoTXYNzSD9f6fF2IvRVHdA",
"UCQSEAbOs6vsJfy7WN7iYaGQ",
"UCI3DTtB-a3fJPjKtQ5kYHfA",
"UC8aG3LDTDwNR1UQhSn9uVrw",
"UCHdMK5Ef2El8KbD3L_WgANg",
"UCn4BNPzJDyxkoBgXRuOFeyA",
"UCIPPMRA040LQr5QPyJEbmXA",
"UCEuN0IauvXNTn6KhP9AVJVw",
"UCzYfz8uibvnB7Yc1LjePi4g",
"UCiSVf-UpLC9rRjAT1qRTW0g",
"UC_cvTMeip9po2hZdF3aBXrA",
"UCke6I9N4KfC968-yRcd5YRg",
"UCJZam2u1G0syq3kyqrCXrNw",
"UCsC7Bac1Jpg28k0BSMEz6ZA",
"UC1bwliGvJogr7cWK0nT2Eag",
"UC7_YxT-KID8kRbqZo7MyscQ",
"UC-lHJZR3Gqxm24_Vd_AJ5Yw",
"UCYzPXprvl5Y-Sf0g4vX-m6g",
"UCNAz5Ut1Swwg6h6ysBtWFog",
"UGGiMCgtLmwNSEUcIspB3tnA",
"UCHIEZ3ZK3KWrSJQhsiFq1DA",
"UCuK2rjMid48EA6XoQPe4PIA",
"UCk8-Skcx2jfBsOtHgmzKKhg",
"UCKhRm8klDao9Nq0aBC0KmBw",
"UCP_1oRZ0RjNwgYmmW-CZieg",
"UC0VVYtw21rg2cokUystu2Dw",
"UCZ9lCUhUOUrwqVJmfBkN92A",
"UCcAbV5NEHxbhdVKHirkoObA",
"UCBXN0W_SEgbcLWjB5ycK5Ug",
"UC_AjfAO4mQWl9215YpVZEog",
"UC-hSZv6tzHEUqrUB6mgXvIA",
"UC-SpacLBhJEHbedPuU5QI3w",
"UC_oCw5PLyGQEJvRtlkjwS6A",
"UCYQ2ZdAFYep5-t91WFEdLog",
"UCPgj-dhrdajrJDP2WQjgPvQ",
"UCsM3qWjsbJr5Kt7_Q1ulLjg",
"UC-9XXmqSEMOBU6bv2ESl7CA",
"UCzCMYNVFeF8wluyfW5_4V8w",
"UC5QgPBgPMM6gSV7hQcHgOhg",
"UC7LEEvPCzPAtHLbQeYuJo6w",
"UCO1YGhO4dFcVxzYBU1oTVdQ",
"UCT7VEYOLH5CxNPZJr54Xp9g",
"UCMwQLdr2C6GTSmPgQEy-lSg",
"UCj0sKQk9qjiIfNH22ZBWpxw",
"UCvzoHz_9Vym9HUJ3lNRfCcw",
"UCxo56gzJQ_fhb6svPqTSewg",
"UCmSfdtegCsyBe8qCnRu-ijw",
"UC279mtSpGdNDRVE_NqSPofA",
"UC6Fr4tBYuLHgPkL2tEkTrSA",
"UCRGfxiszv5dhqRLA-OpD6nQ"]

"""
# Goes through channels, and runs the downloader. The target amount is set to 10k videos, but if the channel has less than that, it stops
for i in tqdm(channels):
    try:
        downloader = channelDownloader(api_key=Api_key,channelID=i, targetAmount=10000)
        downloader.iterateOverChannelPlaylists()
    except Exception as e:
        print(e)
```

## Third Part

The following shows the code used to iterate through the videoID lists for each channel, upload the data to ElasticSearch.

## UploadToES.py


```python
import os
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
from datetime import datetime
from requests import api
from apiclient.discovery import build

from tqdm import tqdm
import requests
import json
import re
import time
import requests


Api_key = open("apikey.txt", 'r')
Api_key = str(Api_key.read())



if os.path.exists('token.pickle'):
    print('Loading Credentials From File...')
    with open('token.pickle', 'rb') as token:
        credentials = pickle.load(token)
        print(credentials.to_json())
else:
    credentials = ""

# This man saved my life:
# https://www.youtube.com/watch?v=th5_9woFJmk
# Bless him
# Corey Schafer



from google.auth.transport.requests import Request
# MAKE SURE REDIRECT IN GOOGLE CLOUD SERVICES CONSOLE IS http://localhost:8080/ WITH THE SLASH AFTERWARDS

# If there are no valid credentials available, then either refresh the token or log in.
if not credentials or not credentials.valid:
    if credentials and credentials.expired and credentials.refresh_token:
        print('Refreshing Access Token...\n\n\n')
        credentials.refresh(Request())
    else:
        print('Fetching New Tokens...\n\n\n')
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret_3.json',
            scopes=[
                'https://www.googleapis.com/auth/youtube.readonly'
            ]
        )

        flow.run_local_server(port=8080, prompt='consent',
                              authorization_prompt_message='')
        credentials = flow.credentials

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as f:
            print('Saving Credentials for Future Use...')
            print(credentials.to_json())
            pickle.dump(credentials.to_json(), f)



class channelUploader:
    def __init__(self, *,api_key, channelID):
        self.api_key = api_key
        self.channelID = channelID
        self.counter = 0
        self.total_requests = 0
        url = "https://www.googleapis.com/youtube/v3/channels?part=snippet%2Cstatus%2CcontentDetails%2Cstatistics%2CbrandingSettings&id=" +  str(self.channelID) + "&key=" + str(self.api_key)
        resp = requests.get(url)
        # print(url)
        data = resp.content
        data = json.loads(data)
        # Wikndows Version
        # w = open(".\\channels.txt","a")
        w = open("./channels.txt","a")
        w.write( str(self.channelID) + "\n")
        w.close()
        # Windows Version
        # os.mkdir(".\\channels\\" + str(self.channelID))
        # os.mkdir("./channels/" + str(self.channelID))
        self.subscriberCount = int(data["items"][0]["statistics"]["subscriberCount"])
        
        moderateComments = False
        madeForKids = False
        country = ""
        if("moderateComments" in  data["items"][0]["brandingSettings"]["channel"]):
            moderateComments = bool(data["items"][0]["brandingSettings"]["channel"]["moderateComments"])
        if("madeForKids" in  data["items"][0]["status"]):
            madeForKids = bool(data["items"][0]["status"]["madeForKids"])
        if("country" in data["items"][0]["brandingSettings"]["channel"]):
            country = str(data["items"][0]["brandingSettings"]["channel"]["country"])
        output = {
            "country" : country,
            "videoCount": int(data["items"][0]["statistics"]["videoCount"]),
            "totalViewCount": int(data["items"][0]["statistics"]["viewCount"]),
            "subscriberCount" : int(data["items"][0]["statistics"]["subscriberCount"]),
            "moderateComments" :moderateComments,
            "madeForKids" : madeForKids,
            "title" : str(data["items"][0]["brandingSettings"]["channel"]["title"]),
            "description" : str(data["items"][0]["snippet"]["description"]),
            "thumbnail" : str(data["items"][0]["snippet"]["thumbnails"]["default"]["url"]),
            "channelInception" : str(data["items"][0]["snippet"]["publishedAt"])
        }
        headers = {

            "Authorization" : "Basic ZG5pZXZlc2FjYXJvMjAxODpGbG9yaWRhX1RlY2hfQGNjZXNzX0RlbkFfMjM3NDMzNjI2NzczNTc=",
            'Content-Type': 'application/json'
        }
        response = requests.post("https://search-mth5320-project1-2-2t7qhhcg23lmtwmbp6m4pwlcya.us-east-2.es.amazonaws.com/channels/_doc", data=json.dumps(output),headers=headers)
        # print("\n\n")
        # print(response.status_code)
        # print(response.content)
        # print("Number of videos is ", self.videoCount)
        # self.nextPageToken = ""
        # print(output)
        # return url

    def getVidInfo(self, *,videoID, uploader, document_id):
        url = "???"
        try:
            url = "https://www.youtube.com/watch?v=" + str(videoID)
            # print(url)
            resp = requests.get(url)
        except Exception as e:
            print(url)
            print(e)
        # print(resp.content)
        # game_title = re.findall(r'/(<div id="title" class="style-scope ytd-rich-metadata-renderer">)[A-z]*</div>/g', str(resp.content))
        # file = open("test.txt" , "r")
        # t = file.read()
        # w = open("curloutput.txt", 'w')
        # w.write(str(resp.content))
        upload_date_indx = str(resp.content).find("<meta itemprop=\"uploadDate\" content=")
        upload_date = str(resp.content)[upload_date_indx + len("<meta itemprop=\"uploadDate\" content=") + 1 :upload_date_indx + len("<meta itemprop=\"uploadDate\" content=") + 12 -1]
        upload_date.replace("\"", "")
        # print(upload_date)
        
        time = datetime(year=int(upload_date[:4]),month=int(upload_date[5:7]),day=int(upload_date[8:]))
        # print("Month is ", time.month)
        # print("Year is ",  time.year)
        # print("Day is ",   time.day)
        time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        upload_date = time

        game_name_indx = str(resp.content).find("}]},\"title\":{\"simpleText\":")
        # game_title = re.findall('}]},"title":{"simpleText":', str(resp.content))
        # print(game_name_indx)
        game_name = str(resp.content)[game_name_indx:game_name_indx+100].replace("}]},\"title\":{\"simpleText\":\"", "")
        # print(game_name)
        game_year = game_name[game_name.find("\"},\"subtitle\":{\"simpleText\":\"") + len("\"},\"subtitle\":{\"simpleText\":\""):].replace("\"},\"callToAction\":{\"","")[:4]
        # in case the game year is legit so that I can turn it into an int without having an exception
        temp = game_name
        
        game_name = game_name[:game_name.find("},\"subtitle\":{")-1]
        if(game_name.find(":{\"")!=-1 or not (game_year.isdigit())):
            game_name="N/A"
            game_year="N/A"
        else:
            game_year = int(temp[temp.find("\"},\"subtitle\":{\"simpleText\":\"") + len("\"},\"subtitle\":{\"simpleText\":\""):].replace("\"},\"callToAction\":{\"","")[:4])
  

        likes_indx = str(resp.content).find("{\"iconType\":\"LIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"") + len("{\"iconType\":\"LIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"")
        likes = str(resp.content)[likes_indx:likes_indx + 50]
        likes = str(likes[:likes.find("likes")])
        likes = likes.strip()
        likes= likes.replace(",", "")
        likes = int(likes)
        # print(url)
        dislike_find_str = "{\"iconType\":\"DISLIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\""
        dislikes_indx = str(resp.content).find(dislike_find_str) + len(dislike_find_str)
        dislikes = str(resp.content)[dislikes_indx:dislikes_indx + 50]
        # print("Dislikes: ",         dislikes)
        # dislikes = str(dislikes[dislikes.find("simpleText\":\"") + len("simpleText\":\""):])
        dislikes = str(dislikes[:dislikes.find("\"")])
        # dislikes = dislikes[:dislikes.find("\"}")]
        # 
        # print("Dislikes: ",         dislikes)
        dislikes = dislikes.strip()
        # print("Dislikes: ",         dislikes)
        dislikes= dislikes.replace(" dislikes", "")
        dislikes= dislikes.replace(" dislike", "")
        dislikes= dislikes.replace(",", "")
        # print("Dislikes: ",         dislikes)
        if(dislikes == "No"):
            dislikes = 0
        else:
            dislikes = int(dislikes)
        
        viewcount_indx = str(resp.content).find("\"viewCount\":{\"videoViewCountRenderer\":{\"viewCount\":{\"simpleText\":\"") + len("\"viewCount\":{\"videoViewCountRenderer\":{\"viewCount\":{\"simpleText\":\"")
        viewcount = str(resp.content)[viewcount_indx:viewcount_indx + 50]
        # print("ViewCount",          viewcount)
        viewcount = str(viewcount[:viewcount.find(" views")])
        # print("ViewCount",          viewcount)
        viewcount = viewcount.strip()
        # print("ViewCount",          viewcount)
        viewcount= viewcount.replace(",", "")
        # print("ViewCount",          viewcount)
        # print(url)
        if(viewcount == "No"):
            viewcount = 0
        else:
            viewcount = int(viewcount)
        
        # print("Game Year of Pub: ", game_year)
        # print("Game Name: ",        game_name)
        # print("Likes: ",            likes)
        # print("ViewCount",          viewcount)
        # print("Dislikes: ",         dislikes)
        # print("ViewCount / SubscriberCount: ",         viewcount / subscriberCount)
        # print("SubscriberCount: ",subscriberCount)
        # print("Upload Date: ", upload_date)
        # print(e)
        # comment_count_indx = str(resp.content).find() + len()
        # comment_count = str(resp.content)[comment_count_indx:comment_count_indx + 50]
        # comment_count = str(comment_count[:comment_count.find(" comment_s")])
        # comment_count = comment_count.strip()
        # comment_count= comment_count.replace(",", "")
        # comment_count = int(comment_count)
        # print(url)
        # print("Game Year of Pub: ", game_year)
        # print("Game Name: ",        game_name)
        # print("Likes: ",            likes)
        # print("ViewCount",          viewcount)
        # print("Dislikes: ",         dislikes)
        # print("ViewCount / SubscriberCount: ",         viewcount / subscriberCount)
        # print("SubscriberCount: ",subscriberCount)
        # print("Upload Date: ", upload_date)
        l_div_d = 0
        l_div_s = 0
        v_div_s = 0
        if(dislikes == 0):
            l_div_d = -99
        else:
            l_div_d = likes/dislikes
        if(self.subscriberCount == 0):
            l_div_s = -99
            v_div_s = -99
        else:
            l_div_s = likes/self.subscriberCount
            v_div_s = viewcount/self.subscriberCount

        output = {
        "id" : videoID,
        "game_year": game_year,
        "game_name":        game_name,
        "likes":        likes,
        "views":          viewcount,
        "dislikes":         dislikes,
        "l/d":         l_div_d,
        "v/s" : v_div_s,
        "l/s" : l_div_s,
        "uploadDate" : upload_date,
        "channel" : self.channelID,
        "uploader" : uploader
        }
        headers = {

            "Authorization" : "Basic ZG5pZXZlc2FjYXJvMjAxODpGbG9yaWRhX1RlY2hfQGNjZXNzX0RlbkFfMjM3NDMzNjI2NzczNTc=",
            'Content-Type': 'application/json'
        }
        response = requests.post("https://search-mth5320-project1-2-2t7qhhcg23lmtwmbp6m4pwlcya.us-east-2.es.amazonaws.com/" + document_id + "/_doc", data=json.dumps(output),headers=headers)


# channel separation for distributing the work
list_1 = [
"UC0VVYtw21rg2cokUystu2Dw",
"UC1bwliGvJogr7cWK0nT2Eag",  
"UC5QgPBgPMM6gSV7hQcHgOhg",  
"UC5v2QgY2D5tlu8uws23MG4Q",  
"UC6Fr4tBYuLHgPkL2tEkTrSA",  
"UC7_YxT-KID8kRbqZo7MyscQ",  
"UC8aG3LDTDwNR1UQhSn9uVrw",  
"UC9PD3EIAA-vtGLZgYXG3q0A",  
"UCam8T03EOFBsNdR0thrFHdQ"
]

list_2 = [
"UCBXN0W_SEgbcLWjB5ycK5Ug",  
"UCcAbV5NEHxbhdVKHirkoObA",  
"UCcG-OmRBrHqiwI91ZbhZt2w",  
"UC-cnbLlplnXA4oc7rR21qzg",  
"UC_cvTMeip9po2hZdF3aBXrA",  
"UCcZUc0Wbt4EPVktbB8FrugQ",  
"UCdKuE7a2QZeHPhDntXVZ91w",  
"UCEPuItFWOOJ2o5hTu65NlEg",  
"UCEuN0IauvXNTn6KhP9AVJVw"
]

list_3 = [
"UCfgh3Ul_dG6plQ7rzuOLx-w",  
"UCFR2oaNj02WnXkOgLH0iqOA",  
"UCHdMK5Ef2El8KbD3L_WgANg",  
"UC-hSZv6tzHEUqrUB6mgXvIA",  
"UCI3DTtB-a3fJPjKtQ5kYHfA",  
"UCIPPMRA040LQr5QPyJEbmXA",  
"UCj0sKQk9qjiIfNH22ZBWpxw",  
"UCj1J3QuIftjOq9iv_rr7Egw",  
"UCjlWEMQ2jbTy_2nWq8sEliw"
]

list_6 = [
"UCK0_slr6cUzYFL_dsiuW7Xw",  
"UCk8-Skcx2jfBsOtHgmzKKhg",  
"UCKQEfXR_uflA4vDtJHmEIIQ",  
"UCkxctb0jr8vwa4Do6c6su0Q",  
"UCL3r1JcBQM1cmyMhMFm6XkQ",  
"UC-lHJZR3Gqxm24_Vd_AJ5Yw",  
"UCn4BNPzJDyxkoBgXRuOFeyA",  
"UCNAz5Ut1Swwg6h6ysBtWFog",  
"UCO1YGhO4dFcVxzYBU1oTVdQ"
]


list_5 = [
"UCoIXnB865l9Ex9zs4OIXTdQ", 
"UCPbGiUt4Yu8EsaFM-KAmgjg", 
"UCPgj-dhrdajrJDP2WQjgPvQ", 
"UCPYJR2EIu0_MJaDeSGwkIVw", 
"UCqg3BHb31w2h77vFuTFmp2w", 
"UCQSEAbOs6vsJfy7WN7iYaGQ", 
"UCRGfxiszv5dhqRLA-OpD6nQ", 
"UCsM3qWjsbJr5Kt7_Q1ulLjg", 
"UC-SpacLBhJEHbedPuU5QI3w"
]


list_7 = [
"UCUdF4kyAKLyT1xYTjddW5_w",
"UCuJyaxv7V-HK4_qQzNK_BXQ",
"UCuK2rjMid48EA6XoQPe4PIA",
"UCVFWJkN7L45x8gZTMXu2UWw",
"UCw1SQ6QRRtfAhrN_cjkrOgA",
"UCWzLmNWhgeh3h1j-M-Isy0g",
"UCXGPGV90SPduyn9LVX9s7Kw",
"UCxo56gzJQ_fhb6svPqTSewg",
"UCyMy3i-BaVOmOwTZskm52Ew"
]


list_8 = [
"UCYQ2ZdAFYep5-t91WFEdLog",
"UCYVinkwSX7szARULgYpvhLw",
"UCYzPXprvl5Y-Sf0g4vX-m6g",
"UCzYfz8uibvnB7Yc1LjePi4g"
]

# Windows Version:
directory = r'C:\\Users\SplashFreeze\Desktop\Semester 7\\MTH 4312\\Project1\\channels'
# directory = '/home/davidn/Documents/MTH-5320/Project1/channels'

failed_channels = []
parallel = []
    # Change list_1                         
    # For each computer you want to distri
    # bute the work in   
for filename in tqdm(list_2):
    # print(os.path.join(directory, filename))
    # try:
    # Windows Version
    o = open(os.path.join(directory, filename) + "\\videos.txt","r")
    # o = open(os.path.join(directory, filename) + "/videos.txt","r")
    # print(filename)
    u = channelUploader(api_key=Api_key, channelID=filename)
    lines = o.readlines()
    # print(lines)
    # lines = [lines.rstrip() for line in lines]
    # # with open()

    # Set this equal to a username or identifier for computer
    # in case your computer screws up and you have to restart the upload process
    uploader = "HP-laptop"
                 
    for count, line in tqdm(enumerate(lines)):
                u.getVidInfo(videoID = line.replace("\n", ""), uploader=uploader,document_id="videos2")
    # except Exception as e:
    #     print("Channel: " + filename, "could not get videos...")
    #     print(e)
    #     failed_channels.append(filename)
print("Failed channels are", failed_channels)

```


```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/dashboard.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_8_0.png)
    


## ElasticSearch Setup

The setup of the ELK stack (sometimes simply referred to as ElasticSearch) was done by simply creating an AWS account and following the instructions required. One caveat is that both from the perspective of a college student, and in the interest of fairness, I only used free tier limits on this ELK setup, which meant using two t3.small instances with 20GiB in total of EBS volumes. The reason I chose ElasticSearch was because of the powerful visualizations it provides in the form of Kibana. Here are a few examples:


```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/top_15.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_10_0.png)
    



```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/top_50.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_11_0.png)
    



```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/unique.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_12_0.png)
    



```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/tabl1.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_13_0.png)
    



```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/tabl2.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_14_0.png)
    


## Benchmarking

In order to speed up the process of training, I downloaded a copy of the data locally by querying ElasticSearch. One might think that this defeats the point of having a database, but the true value of ElasticSearch lies in its searching capabilities, its visualizations, as well as the fact that I have all the data in one easily accessible cloud-hosted location where I only have to collect the data once from YouTube and be done with it.


```python
import json
import os
import math
import pickle
from datetime import datetime
from requests import api
import numpy as np
from tqdm import tqdm
import requests
import json
import re
import time
import requests
import ast
import codecs


j = open("sample_videos.json", "r",encoding="utf-8")
j = j.read()
j = codecs.decode(j, "unicode-escape")
w = json.loads(j)
w = [x["_source"] for x in w]
# print(w[0])
seen = dict()


a = []
for elem in w:
    if elem['id'] in seen:
        continue
    else:
        seen[elem['id']] = True
        a.append(elem)
print(a[:2])
```

    [{"id": "VE4VbuV2jVU", "game_year": 2017, "game_name": "Rime", "likes": 3961, "views": 57890, "dislikes": 48, "l/d": 82.52083333333333, "v/s": 0.04256617647058823, "uploadDate": "2017-06-06T00:00:00Z", "channel": "UCXGPGV90SPduyn9LVX9s7Kw"}, {"id": "zrtdVAE1EK8", "game_year": 2017, "game_name": "Arms", "likes": 16851, "views": 567993, "dislikes": 321, "l/d": 52.495327102803735, "v/s": 0.4176419117647059, "uploadDate": "2017-06-06T00:00:00Z", "channel": "UCXGPGV90SPduyn9LVX9s7Kw"}]
    

For inputting one of the most important features, I made a function which generates a one-hot encoding of which game is being played in the video. Note that I only used the top 15 categories, including the "N/A" category, which includes off-topic videos which might talk about video games or something video game related, but might not necessarily have a video game entry in the video's page, as seen below:


```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/ex_entry.png")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_18_0.png)
    



```python

# This list was made with the help of Kibana
top_15 = ["N/A",
"Minecraft",
"Fortnite",
"Grand Theft Auto V",
"Roblox",
"Among Us",
"Happy Wheels",
"Five Nights at Freddys",
"PUBG MOBILE",
"Garrys Mod",
"Tom Clancys Rainbow Six Siege",
"Call of Duty: Black Ops III",
"Subnautica",
"Call of Duty: Advanced Warfare",
"Hello Neighbor"]

def one_hot_games(game, unique_games):
    # automatically adds an "other" category to mark for the non-top 15 in our case
    a = [0]*(len(unique_games) + 1)
    try:
        a[unique_games.index(game)] = 1
    except Exception as e:
        # if it is not in any, simply add a 1 to the "other" category
        a[-1] = 1
    return a
```

As for processing and normalizing the inputs, I decided to go firsthand by taking several ratios, such as 

```"l/d"```  $\frac{likes}{dislikes}$, 

```"v/s"``` or $\frac{views}{subscribers}$,

 as well as a non-ratio such as the raw number of subscribers into a power and then dividing that by the max value in order to normalize. I made sure that my values were sensical by printing out the entry to which the max value belonged to. For example, the max subscribers in this whole list belongs to the user PewDiePie, which makes sense. 


```python




l_div_d = [math.log(x["l/d"]) if x["l/d"] >0 else 0.1  for x in a]
max_l_div_d = l_div_d[np.argmax(l_div_d)]
print("Max l_div_d is ","https://youtube.com/watch?v=" +str(a[np.argmax(l_div_d)]["id"]))
print("Belongs to ", a[np.argmax(l_div_d)]["id"])
v_div_s = [math.log(x["v/s"]) if x["l/d"] > 0 else 0.1 for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))

v_div_s = [math.log(x["v/s"]) for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))


subscribers = [math.log(max(0.1, x["views"]/x["v/s"])) for x in a]
max_subs = subscribers[np.argmax(subscribers)]
print("Max subscribers is ", max_subs)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(subscribers)]["id"]))



# np.argmax(])
inputs = [([math.log(max(0.1, x["l/d"]))/max_l_div_d,math.log(max(0,x["v/s"]))/max_v_div_s,math.log(max(0.1,x["views"]/x["v/s"]))/max_subs ] +one_hot_games(x["game_name"], top_15)) for x in a]
print("Ex. First entry of inputs is ")
print(inputs[0])
inputs = np.array(inputs,dtype="float")
# print(inputs[0].shape)

outputs = [[min(4,max(0, math.log( x["views"], 10)-3))] for x in a]  
outputs = np.array(outputs,dtype="float")
# print(outputs.shape)

```

    Max l_div_d is  https://youtube.com/watch?v=THbdeQrnHm0
    Belongs to  THbdeQrnHm0
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max subscribers is  18.51599092375669
    Belongs to  https://youtube.com/watch?v=YlXFDUL6Qa4
    Ex. First entry of inputs is 
    [0.6819946722401631, -0.9511739742239801, 0.7627458511870363, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    

Then, I try using a benchmark of a one-layer neural net (essentially multi-class linear regression) to serve as a benchmark of how the more complicated architecture of a neural network along with its associated hyperparameters improves the results.


```python

# load some more libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(linewidth=180)
class FeedforwardNeuralNetworkSGD:
    
    def __init__(self, layers, alpha = 0.1, batchSize = 32):
        self.W = []
        
        self.layers = layers
        
        self.alpha = alpha
        # Change in Activation function!
        # C = 2.5
        self.c = 2.5
        derivatice_elu_single = lambda x: (self.c * np.exp(x)) if(x <=0) else 1
        self.vectorized_ELU_Derivatice= np.vectorize(derivatice_elu_single)
        elu_single = lambda x: (self.c * np.exp(x) - 1) if(x <=0) else x
        self.vectorized_ELU = np.vectorize(elu_single)

        
        
        self.batchSize = batchSize
        
        
        for i in np.arange(0, len(layers) - 2):
            self.W.append(np.random.randn(layers[i] + 1, layers[i + 1] + 1)/100.0)
            
        self.W.append(np.random.randn(layers[-2] + 1, layers[-1])/100.0)
        
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, z):
        return z * (1 - z)
    
    def getNextBatch(self, X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])
    
    def fit(self, X, y, epochs = 10000, update = 1000):
        X = np.hstack((X, np.ones([X.shape[0],1])))

        for epoch in tqdm(np.arange(0,epochs)):
            
            p = np.arange(0,X.shape[0])
            np.random.shuffle(p)
            X = X[p]
            y = y[p]

            for (x, target) in self.getNextBatch(X, y, self.batchSize):
                A = [np.atleast_2d(x)]
                B = [np.atleast_2d(x)]
                
                for layer in np.arange(0, len(self.W)):
                    
                    net = A[layer].dot(self.W[layer])
                    out = self.vectorized_ELU(net)
                    
                    A.append(out)
                    B.append(net)
                    
                error = A[-1] - target
                D = [error * self.vectorized_ELU_Derivatice(B[-1])]
                
                for layer in np.arange(len(A) - 2, 0, -1):
                    delta = D[-1].dot(self.W[layer].T)
                    delta = delta * self.vectorized_ELU_Derivatice(B[layer])
                    D.append(delta)
                    
                D = D[::-1]
                
                for layer in np.arange(0, len(self.W)):
                    self.W[layer] -= self.alpha * A[layer].T.dot(D[layer])
                    
                
    def predict(self, X, addOnes = True):
        p = np.atleast_2d(X)
        
        if addOnes:
            p = np.hstack((p, np.ones([X.shape[0],1])))
        
        for layer in np.arange(0, len(self.W)):
            p = self.vectorized_ELU(np.dot(p, self.W[layer]))
            
        return p
    
    def computeLoss(self, X, y):
        y = np.atleast_2d(y)
        
        predictions = self.predict(X, addOnes = False)
        loss = np.sum((predictions - y)**2) / 2.0
        
        return loss





np.random.seed(7468378)
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)


# trainX = trainX.astype('float64')/255.0
# testX = testX.astype('float64')/255.0

# trainX = trainX.reshape([60000, 28*28])
# testX = testX.reshape([10000, 28*28])

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],testY.shape[1]], 0.005, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:10<00:00,  9.26it/s]
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.98      0.55      0.71       206
               1       0.72      0.47      0.57       809
               2       0.75      0.79      0.77      2070
               3       0.75      0.93      0.83      1441
               4       0.00      0.00      0.00        94
    
        accuracy                           0.75      4620
       macro avg       0.64      0.55      0.58      4620
    weighted avg       0.74      0.75      0.73      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.97      0.54      0.70       212
               1       0.69      0.45      0.55       833
               2       0.73      0.78      0.76      1999
               3       0.75      0.94      0.83      1456
               4       0.00      0.00      0.00       119
    
        accuracy                           0.74      4619
       macro avg       0.63      0.54      0.57      4619
    weighted avg       0.72      0.74      0.72      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

As can be seen, the benchmark already does decently on the dataset, but we will see that adding more layers (thus turning it from linear regression into the realm of neural networks) improves the results.


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],16,16,testY.shape[1]], 0.005, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:58<00:00,  1.70it/s]
    

    Training set accuracy
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
               0       0.82      0.84      0.83       206
               1       0.70      0.60      0.65       809
               2       0.73      0.89      0.80      2070
               3       0.88      0.73      0.80      1441
               4       0.00      0.00      0.00        94
    
        accuracy                           0.77      4620
       macro avg       0.63      0.61      0.62      4620
    weighted avg       0.76      0.77      0.76      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.83      0.78      0.80       212
               1       0.73      0.60      0.66       833
               2       0.72      0.90      0.80      1999
               3       0.88      0.74      0.80      1456
               4       0.00      0.00      0.00       119
    
        accuracy                           0.77      4619
       macro avg       0.63      0.60      0.61      4619
    weighted avg       0.76      0.77      0.75      4619
    
    

For now we will perform some experiments with the learning rate. Perhaps a more aggressive learning rate of $\alpha = 0.1$ will result in more accurate results. 


```python

(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)


devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],16,16,testY.shape[1]], 0.1, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:51<00:00,  1.95it/s]
    

    Training set accuracy
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
               0       0.05      1.00      0.09       218
               1       0.00      0.00      0.00       839
               2       0.00      0.00      0.00      2060
               3       0.00      0.00      0.00      1413
               4       0.00      0.00      0.00        90
    
        accuracy                           0.05      4620
       macro avg       0.01      0.20      0.02      4620
    weighted avg       0.00      0.05      0.00      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.04      1.00      0.08       200
               1       0.00      0.00      0.00       803
               2       0.00      0.00      0.00      2009
               3       0.00      0.00      0.00      1484
               4       0.00      0.00      0.00       123
    
        accuracy                           0.04      4619
       macro avg       0.01      0.20      0.02      4619
    weighted avg       0.00      0.04      0.00      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Based on this result as well as previous results from other assignments, I will continue to use 0.005 or even smaller gradients. Let me try 0.001.


```python

(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)


devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],16,16,testY.shape[1]], 0.001, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:34<00:00,  2.90it/s]
    

    Training set accuracy
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
               0       0.73      0.28      0.40       219
               1       0.62      0.69      0.65       831
               2       0.70      0.74      0.72      2000
               3       0.73      0.74      0.74      1475
               4       0.00      0.00      0.00        95
    
        accuracy                           0.69      4620
       macro avg       0.56      0.49      0.50      4620
    weighted avg       0.68      0.69      0.68      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.75      0.32      0.45       199
               1       0.64      0.67      0.65       811
               2       0.71      0.74      0.73      2069
               3       0.69      0.75      0.72      1422
               4       0.00      0.00      0.00       118
    
        accuracy                           0.69      4619
       macro avg       0.56      0.50      0.51      4619
    weighted avg       0.68      0.69      0.68      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

This one did about the same as $\alpha = 0.005$ but given its inferior results with respect to the 4 classification, I will proceed with $\alpha = 0.005$. Next I will try varying the network architecture by doubling the size of the first hidden layer. With any luck, this should be able to predict more accurately.


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.005, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [01:34<00:00,  1.06it/s]
    

    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.87      0.94      0.91       218
               1       0.78      0.81      0.79       802
               2       0.75      0.85      0.80      2025
               3       0.86      0.69      0.76      1474
               4       0.95      0.60      0.74       101
    
        accuracy                           0.79      4620
       macro avg       0.84      0.78      0.80      4620
    weighted avg       0.80      0.79      0.79      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.83      0.92      0.87       200
               1       0.80      0.80      0.80       840
               2       0.75      0.87      0.81      2044
               3       0.87      0.69      0.77      1423
               4       0.96      0.66      0.78       112
    
        accuracy                           0.80      4619
       macro avg       0.84      0.79      0.81      4619
    weighted avg       0.81      0.80      0.80      4619
    
    

That only did slightly better than the input, 16, 16, output configuration seen above. Perhaps decreasing the batch size can help?


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.005, 4)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [01:23<00:00,  1.19it/s]
    

    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.90      0.60      0.72       184
               1       0.87      0.54      0.66       858
               2       0.68      0.95      0.79      2029
               3       0.90      0.64      0.75      1449
               4       0.44      0.49      0.46       100
    
        accuracy                           0.75      4620
       macro avg       0.76      0.64      0.68      4620
    weighted avg       0.79      0.75      0.74      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.92      0.62      0.74       234
               1       0.87      0.60      0.71       784
               2       0.69      0.95      0.80      2040
               3       0.88      0.60      0.72      1448
               4       0.43      0.44      0.44       113
    
        accuracy                           0.75      4619
       macro avg       0.76      0.64      0.68      4619
    weighted avg       0.78      0.75      0.75      4619
    
    

Perhaps a happy medium of batch size = 8 can be found?


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.005, 8)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [01:14<00:00,  1.33it/s]
    

    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.98      0.79      0.88       212
               1       0.71      0.77      0.74       780
               2       0.76      0.82      0.79      2036
               3       0.81      0.67      0.73      1475
               4       0.27      0.44      0.34       117
    
        accuracy                           0.75      4620
       macro avg       0.71      0.70      0.70      4620
    weighted avg       0.77      0.75      0.76      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.99      0.84      0.91       206
               1       0.76      0.77      0.77       862
               2       0.76      0.82      0.79      2033
               3       0.81      0.70      0.75      1422
               4       0.25      0.41      0.31        96
    
        accuracy                           0.77      4619
       macro avg       0.71      0.71      0.71      4619
    weighted avg       0.78      0.77      0.77      4619
    
    

Clearly, the batch size of 16 does better. Perhaps increasing the batch size to 32 can yield better results?


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.005, 32)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [01:24<00:00,  1.19it/s]
    

    Training set accuracy
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
               0       0.04      1.00      0.08       198
               1       0.00      0.00      0.00       821
               2       0.00      0.00      0.00      2008
               3       0.00      0.00      0.00      1483
               4       0.00      0.00      0.00       110
    
        accuracy                           0.04      4620
       macro avg       0.01      0.20      0.02      4620
    weighted avg       0.00      0.04      0.00      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.05      1.00      0.09       220
               1       0.00      0.00      0.00       821
               2       0.00      0.00      0.00      2061
               3       0.00      0.00      0.00      1414
               4       0.00      0.00      0.00       103
    
        accuracy                           0.05      4619
       macro avg       0.01      0.20      0.02      4619
    weighted avg       0.00      0.05      0.00      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

This probably has resulted in the gradient exploding, so the learning rate will be reduced.


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.0005, 32)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:24<00:00,  4.05it/s]
    

    Training set accuracy
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       211
               1       0.51      0.38      0.43       808
               2       0.48      0.78      0.59      2030
               3       0.52      0.26      0.35      1464
               4       0.00      0.00      0.00       107
    
        accuracy                           0.49      4620
       macro avg       0.30      0.28      0.28      4620
    weighted avg       0.47      0.49      0.45      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       207
               1       0.52      0.39      0.45       834
               2       0.49      0.79      0.61      2039
               3       0.53      0.26      0.35      1433
               4       0.00      0.00      0.00       106
    
        accuracy                           0.50      4619
       macro avg       0.31      0.29      0.28      4619
    weighted avg       0.47      0.50      0.46      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Using an even lower learning rate to further compensate for the large gradient...


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.00005, 32)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    

    100%|██████████| 100/100 [00:20<00:00,  4.99it/s]
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       207
               1       0.00      0.00      0.00       832
               2       0.44      1.00      0.61      2039
               3       0.00      0.00      0.00      1441
               4       0.00      0.00      0.00       101
    
        accuracy                           0.44      4620
       macro avg       0.09      0.20      0.12      4620
    weighted avg       0.19      0.44      0.27      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       211
               1       0.00      0.00      0.00       810
               2       0.44      1.00      0.61      2030
               3       0.00      0.00      0.00      1456
               4       0.00      0.00      0.00       112
    
        accuracy                           0.44      4619
       macro avg       0.09      0.20      0.12      4619
    weighted avg       0.19      0.44      0.27      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Based on this, and on the dimishing returns, it seems safe to say that the batch size of 16 should be the best for what is necessary here. Next I will try using the sigmoid derivative instead of ReLU to see if better results can be achieved.


```python
class FeedforwardNeuralNetworkSGD:
    def __init__(self, layers, alpha = 0.1, batchSize = 32):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.batchSize = batchSize

        for i in np.arange(0, len(layers) - 2):
            self.W.append(np.random.randn(layers[i] + 1, layers[i + 1] + 1))
            
        self.W.append(np.random.randn(layers[-2] + 1, layers[-1]))
        
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, z):
        return z * (1 - z)
    
    def getNextBatch(self, X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])
    
    def fit(self, X, y, epochs = 10000, update = 1000):
        X = np.hstack((X, np.ones([X.shape[0],1])))

        for epoch in np.arange(0,epochs):
            p = np.arange(0,X.shape[0])
            np.random.shuffle(p)
            X = X[p]
            y = y[p]

            for (x, target) in self.getNextBatch(X, y, self.batchSize):
                A = [np.atleast_2d(x)]
                
                for layer in np.arange(0, len(self.W)):
                    net = A[layer].dot(self.W[layer])
                    out = self.sigmoid(net)
                    A.append(out)

                error = A[-1] - target
                
                D = [error * self.sigmoidDerivative(A[-1])]
                for layer in np.arange(len(A) - 2, 0, -1):
                    delta = D[-1].dot(self.W[layer].T)
                    delta = delta * self.sigmoidDerivative(A[layer])
                    D.append(delta)
                    
                D = D[::-1]
                
                for layer in np.arange(0, len(self.W)):
                    self.W[layer] -= self.alpha * A[layer].T.dot(D[layer])
                    
            if (epoch + 1) % update == 0:
                loss = self.computeLoss(X,y)
                print('Epoch =', epoch + 1, '\t loss =', loss)
                
    def predict(self, X, addOnes = True):
        p = np.atleast_2d(X)
        
        if addOnes:
            p = np.hstack((p, np.ones([X.shape[0],1])))
        
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
        return p
    
    def computeLoss(self, X, y):
        y = np.atleast_2d(y)
        
        predictions = self.predict(X, addOnes = False)
        loss = np.sum((predictions - y)**2) / 2.0
        
        return loss
```


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([devX.shape[1],32,16,testY.shape[1]], 0.005, 16)
model.fit(devX, devY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(devX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
print(classification_report(devY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    Epoch = 10 	 loss = 1403.2865075239638
    Epoch = 20 	 loss = 1295.7698053970717
    Epoch = 30 	 loss = 1192.7432920298759
    Epoch = 40 	 loss = 1078.846815204404
    Epoch = 50 	 loss = 963.7294660024506
    Epoch = 60 	 loss = 857.6145384685033
    Epoch = 70 	 loss = 762.6575790024502
    Epoch = 80 	 loss = 672.1129920592439
    Epoch = 90 	 loss = 593.2913987115348
    Epoch = 100 	 loss = 527.9205566488499
    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.99      0.62      0.77       202
               1       0.86      0.76      0.80       834
               2       0.88      0.97      0.92      2045
               3       0.90      0.96      0.93      1424
               4       0.00      0.00      0.00       115
    
        accuracy                           0.89      4620
       macro avg       0.73      0.66      0.68      4620
    weighted avg       0.87      0.89      0.87      4620
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.99      0.55      0.71       216
               1       0.83      0.78      0.81       808
               2       0.89      0.97      0.93      2024
               3       0.91      0.96      0.94      1473
               4       0.00      0.00      0.00        98
    
        accuracy                           0.89      4619
       macro avg       0.73      0.65      0.68      4619
    weighted avg       0.88      0.89      0.88      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

This is speculation, but it seems like it could be the case that the reason why it does better with sigmoid is because of the fact that the activation for sigmoid makes use of $e$, and that many of the inputs are logarithmic. 

Finally, wrapping things up and using the training set on this last setup, we get:



```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

trainY = to_categorical(trainY, 5)
devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([trainX.shape[1],32,16,trainY.shape[1]], 0.005, 16)
model.fit(trainX, trainY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(trainX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
trainY = trainY.argmax(axis=1)
print(classification_report(trainY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 19)
    Outputs shape is  (4619, 5)
    Epoch = 10 	 loss = 4831.3394595768095
    Epoch = 20 	 loss = 2637.7142878897885
    Epoch = 30 	 loss = 2120.8178476481594
    Epoch = 40 	 loss = 1535.1877282946155
    Epoch = 50 	 loss = 1315.9008743506008
    Epoch = 60 	 loss = 1284.4725078167007
    Epoch = 70 	 loss = 1331.1564977827568
    Epoch = 80 	 loss = 1055.6417006823713
    Epoch = 90 	 loss = 934.3459501095784
    Epoch = 100 	 loss = 973.1230600022394
    Training set accuracy
                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98      1662
               1       0.99      0.97      0.98      6878
               2       0.99      0.98      0.98     16159
               3       0.91      1.00      0.95     11482
               4       0.00      0.00      0.00       774
    
        accuracy                           0.96     36955
       macro avg       0.78      0.78      0.78     36955
    weighted avg       0.95      0.96      0.95     36955
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.99      0.96      0.98       188
               1       0.99      0.96      0.98       836
               2       0.98      0.97      0.98      2004
               3       0.90      1.00      0.95      1484
               4       0.00      0.00      0.00       107
    
        accuracy                           0.96      4619
       macro avg       0.77      0.78      0.78      4619
    weighted avg       0.94      0.96      0.95      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Thus, we get pretty good results in the end. The pattern provided is probably due to the neural network sort of figuring out that by multiplying the views/subscriber ratio times the subscribers (or rather, by adding since they are logarithms), one can get a sort of "baseline" amount of views, given that most subscribers, especially for big channels, will watch the video. Thus, if a channel has about 1M subscribers, and a view/subscriber ratio of about 1.5, it can likely predict that the video will easily get about 1.5M views as a sort of baseline (maybe more, maybe less). Of course, this is only speculation. 

One final attempt: trying out the inference without inputting the viewer/subscriber ratio.


```python

l_div_d = [math.log(x["l/d"]) if x["l/d"] >0 else 0.1  for x in a]
max_l_div_d = l_div_d[np.argmax(l_div_d)]
print("Max l_div_d is ","https://youtube.com/watch?v=" +str(a[np.argmax(l_div_d)]["id"]))
print("Belongs to ", a[np.argmax(l_div_d)]["id"])
v_div_s = [math.log(x["v/s"]) if x["l/d"] > 0 else 0.1 for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))

v_div_s = [math.log(x["v/s"]) for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))


subscribers = [math.log(max(0.1, x["views"]/x["v/s"])) for x in a]
max_subs = subscribers[np.argmax(subscribers)]
print("Max subscribers is ", max_subs)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(subscribers)]["id"]))



# np.argmax(])
inputs = [([math.log(max(0.1, x["l/d"]))/max_l_div_d,math.log(max(0.1,x["views"]/x["v/s"]))/max_subs ] +one_hot_games(x["game_name"], top_15)) for x in a]
print("Ex. First entry of inputs is ")
print(inputs[0])
inputs = np.array(inputs,dtype="float")
# print(inputs[0].shape)

outputs = [[min(4,max(0, math.log( x["views"], 10)-3))] for x in a]  
outputs = np.array(outputs,dtype="float")
# print(outputs.shape)
```

    Max l_div_d is  https://youtube.com/watch?v=THbdeQrnHm0
    Belongs to  THbdeQrnHm0
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max subscribers is  18.51599092375669
    Belongs to  https://youtube.com/watch?v=YlXFDUL6Qa4
    Ex. First entry of inputs is 
    [0.6819946722401631, 0.7627458511870363, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    


```python
(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

trainY = to_categorical(trainY, 5)
devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([trainX.shape[1],32,16,trainY.shape[1]], 0.005, 16)
model.fit(trainX, trainY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(trainX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
trainY = trainY.argmax(axis=1)
print(classification_report(trainY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Inputs shape is  (46194, 18)
    Outputs shape is  (4619, 5)
    Epoch = 10 	 loss = 9984.414919701177
    Epoch = 20 	 loss = 9294.052507234843
    Epoch = 30 	 loss = 9069.031433933484
    Epoch = 40 	 loss = 8949.000720151233
    Epoch = 50 	 loss = 8864.384340587167
    Epoch = 60 	 loss = 8850.896952430046
    Epoch = 70 	 loss = 8870.05847859579
    Epoch = 80 	 loss = 8783.74790760452
    Epoch = 90 	 loss = 8758.636937630668
    Epoch = 100 	 loss = 8731.425988549305
    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.98      0.80      0.88      1662
               1       0.61      0.26      0.37      6878
               2       0.60      0.76      0.67     16159
               3       0.68      0.74      0.71     11482
               4       0.00      0.00      0.00       774
    
        accuracy                           0.64     36955
       macro avg       0.57      0.51      0.53     36955
    weighted avg       0.63      0.64      0.62     36955
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.99      0.80      0.88       215
               1       0.56      0.23      0.32       809
               2       0.61      0.76      0.68      2051
               3       0.69      0.74      0.71      1447
               4       0.00      0.00      0.00        97
    
        accuracy                           0.65      4619
       macro avg       0.57      0.51      0.52      4619
    weighted avg       0.63      0.65      0.62      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Now let us try it without an approximation of the subscribers.


```python

l_div_d = [math.log(x["l/d"]) if x["l/d"] >0 else 0.1  for x in a]
max_l_div_d = l_div_d[np.argmax(l_div_d)]
print("Max l_div_d is ","https://youtube.com/watch?v=" +str(a[np.argmax(l_div_d)]["id"]))
print("Belongs to ", a[np.argmax(l_div_d)]["id"])
v_div_s = [math.log(x["v/s"]) if x["l/d"] > 0 else 0.1 for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))

v_div_s = [math.log(x["v/s"]) for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))


subscribers = [math.log(max(0.1, x["views"]/x["v/s"])) for x in a]
max_subs = subscribers[np.argmax(subscribers)]
print("Max subscribers is ", max_subs)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(subscribers)]["id"]))



# np.argmax(])
inputs = [([math.log(max(0.1, x["l/d"]))/max_l_div_d, ] +one_hot_games(x["game_name"], top_15)) for x in a]
print("Ex. First entry of inputs is ")
print(inputs[0])
inputs = np.array(inputs,dtype="float")
# print(inputs[0].shape)

outputs = [[min(4,max(0, math.log( x["views"], 10)-3))] for x in a]  
outputs = np.array(outputs,dtype="float")
# print(outputs.shape)

(trainX, restX, trainY, restY) = train_test_split(inputs, outputs, test_size = 0.20, random_state = 1)

(testX, devX, testY, devY) = train_test_split(restX, restY, test_size = 0.50)

trainY = to_categorical(trainY, 5)
devY = to_categorical(devY, 5)
testY = to_categorical(testY, 5)

print("Inputs shape is ",inputs.shape )
print("Outputs shape is ",testY.shape)
model = FeedforwardNeuralNetworkSGD([trainX.shape[1],32,16,trainY.shape[1]], 0.005, 16)
model.fit(trainX, trainY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(trainX)
predictedY = predictedY.argmax(axis=1)

devY = devY.argmax(axis=1)
trainY = trainY.argmax(axis=1)
print(classification_report(trainY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))
```

    Max l_div_d is  https://youtube.com/watch?v=THbdeQrnHm0
    Belongs to  THbdeQrnHm0
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max v_dix_s is  3.3187360106384216
    Belongs to  https://youtube.com/watch?v=bSN27iYjcdc
    Max subscribers is  18.51599092375669
    Belongs to  https://youtube.com/watch?v=YlXFDUL6Qa4
    Ex. First entry of inputs is 
    [0.6819946722401631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    Inputs shape is  (46194, 17)
    Outputs shape is  (4619, 5)
    Epoch = 10 	 loss = 11361.346416075183
    Epoch = 20 	 loss = 11270.641485508662
    Epoch = 30 	 loss = 11186.275952712662
    Epoch = 40 	 loss = 11096.464327285277
    Epoch = 50 	 loss = 11064.623439400631
    Epoch = 60 	 loss = 11054.835556300255
    Epoch = 70 	 loss = 11033.071755841504
    Epoch = 80 	 loss = 11018.37397764015
    Epoch = 90 	 loss = 11011.3976895489
    Epoch = 100 	 loss = 11079.102874881251
    Training set accuracy
                  precision    recall  f1-score   support
    
               0       0.87      0.52      0.65      1662
               1       0.00      0.00      0.00      6878
               2       0.51      0.75      0.61     16159
               3       0.53      0.55      0.54     11482
               4       0.00      0.00      0.00       774
    
        accuracy                           0.52     36955
       macro avg       0.38      0.36      0.36     36955
    weighted avg       0.43      0.52      0.46     36955
    
    Test set accuracy
                  precision    recall  f1-score   support
    
               0       0.90      0.51      0.65       220
               1       0.00      0.00      0.00       817
               2       0.51      0.75      0.61      2043
               3       0.52      0.55      0.54      1431
               4       0.00      0.00      0.00       108
    
        accuracy                           0.53      4619
       macro avg       0.39      0.36      0.36      4619
    weighted avg       0.43      0.53      0.47      4619
    
    

    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Python39\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

As can be seen, the model definitely suffers quite a bit when removing the viewer/subscriber ratio and the subscribers, which makes sense. This way, it feels like a more accurate representation of what the model could realistically do before knowing the viewer/subscriber ratio beforehand. Of course, one could argue that in real life, the viewer subscriber ratio or an estimate thereof could be supplied from an average of earlier videos by a content creator in order to estimate how many views a next video might receive. 

## Conclusion

The best results were obtained with an accuracy of 0.96, but this was when feeding the model the viewer/subscriber ratio, which although not unrealistic (it can be estimated from previous videos), is still a bit unfair. Some more realistic results without the use of the viewer/subscriber ratio are shown above.

Like I mentioned before, the focus of this project was data collection, which is why I would like to build on top of the effort realized here in order to be able to make more accurate predictions using more important features such as thumbnails, title content, etc... using the more sophisticated neural network types talked about later in the course. 

One key note to consider about my approach to views is that it is rather naive in that it does not take into account the role of the YouTube algorithm. What this means is that while generally, videos with interesting or eye catching thumbnails, titles, and channels willl be more likely to get views, there is still an element of spontaneity that is present when taking into account the YouTube algorithm. This is especailly true when one considers the changes that have been made to it over the years in order to satisfy different objectives such as:

- [Optimizing Watch Time](https://blog.youtube/news-and-events/youtube-now-why-we-focus-on-watch-time/)
- [Optimizing for user satisfaction using neural networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

[Although not an algorithm change, further regulation has been introduced as part of an effort to reduce radicalization and borderline content.](https://blog.youtube/inside-youtube/letter-from-susan-our-2021-priorities/)

## Thank you section

I would like to give credit to Corey Schafer for his [helpful guide](https://www.youtube.com/watch?v=th5_9woFJmk) and code sample on how to use the YouTube Data API in Python (specifically for generating the access code).

## Collection process

Picture of the computers used at Evans Hall to facilitate the collection process.




```python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("images/collection.jpeg")
plt.figure(figsize=(200,100))
plt.imshow(image)
plt.show()

```


    
![png](FinalReport_files/FinalReport_56_0.png)
    

