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

# Google's Request
from google.auth.transport.requests import Request
# MAKE SURE REDIRECT URI IS http://localhost:8080/ WITH THE SLASH AFTERWARDS

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

            "Authorization" : "Basic ",
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

            "Authorization" : "Basic ",
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




# curl -X POST -u 