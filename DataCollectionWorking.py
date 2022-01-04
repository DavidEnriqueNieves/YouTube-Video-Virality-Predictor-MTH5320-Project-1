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



if os.path.exists('token.pickle'):
    print('Loading Credentials From File...')
    with open('token.pickle', 'rb') as token:
        credentials = pickle.load(token)
        print(credentials.to_json())
else:
    credentials = ""

# Google's Request
from google.auth.transport.requests import Request
# MAKE SURE REDIRECT URI IS http://localhost:8080/ WITH THE SLASH AFTERWARDS

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
                videos_request = youtube.playlistItems().list(part="status,id,snippet,contentDetails", playlistId=self.channelID.replace(self.channelID[1], "U"),pageToken=self.nextPageToken, maxResults=50)
                response_videos = videos_request.execute()
                self.nextPageToken=response_videos["nextPageToken"]
            # print("Length of items is ", response_videos["pageInfo"]["resultsPerPage"])
            for video_count, video in enumerate(response_videos["items"]):
                # print("VIDEO COUNT",(i)*50 + video_count  )
                # print(str(video["contentDetails"]["videoId"]))
                w.write(str(video["contentDetails"]["videoId"]) + "\n")
                # print(str(video["contentDetails"]["videoPublishedAt"]))
                # print("VIDEO ","------------------------------------")
                if((i)*50 + video_count >= self.targetAmount - 1 or (i)*50 + video_count >= self.videoCount):
                    w.close()
                    return videos
        w.close()
            
            
        
                



youtube = build("youtube", "v3", credentials = credentials)




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

for i in tqdm(channels):
    try:
        downloader = channelDownloader(api_key=Api_key,channelID=i, targetAmount=10000)
        downloader.iterateOverChannelPlaylists()
    except Exception as e:
        print(e)








