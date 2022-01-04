import requests
import json
import re
import time
Api_key = open("apikey.txt", 'r')
Api_key = str(Api_key.read())



class channelDownloader:
    def __init__(self, *,api_key, channelID, targetAmount):
        self.api_key = api_key
        self.channelID = channelID
        self.targetAmount = targetAmount
        self.counter = 0
        self.total_requests = 0
        url = "https://www.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id=" +  str(self.channelID) + "&key=" + str(self.api_key)
        resp = requests.get(url)
        data = resp.content
        data = json.loads(data)
        self.subscriberCount = int(data["items"][0]["statistics"]["subscriberCount"])
        self.videoCount = int(data["items"][0]["statistics"]["videoCount"])
        print("Number of videos is ", self.videoCount)
        self.nextPageToken = ""

    def getVidInfo(self, videoID):
        url = "https://www.googleapis.com/youtube/v3/videos?id=" + str(videoID) + "&key="+ str(self.api_key) + "&part=snippet,statistics,status"
        resp = requests.get(url)
        # print(url)
        self.total_requests+=1
        data = resp.content
        data = json.loads(data)
        # print(data)

        url = "https://www.youtube.com/watch?v=" + str(videoID)
        # print(url)
        resp = requests.get(url)
        # print(resp.content)
        # game_title = re.findall(r'/(<div id="title" class="style-scope ytd-rich-metadata-renderer">)[A-z]*</div>/g', str(resp.content))
        # file = open("test.txt" , "r")
        # t = file.read()
        w = open("curloutput.txt", 'w')
        w.write(str(resp.content))
        game_name_indx = str(resp.content).find("}]},\"title\":{\"simpleText\":")
        # game_title = re.findall('}]},"title":{"simpleText":', str(resp.content))
        # print(game_name_indx)
        game_name = str(resp.content)[game_name_indx:game_name_indx+100].replace("}]},\"title\":{\"simpleText\":\"", "")
        # print(game_name)
        game_year = game_name[game_name.find("\"},\"subtitle\":{\"simpleText\":\"") + len("\"},\"subtitle\":{\"simpleText\":\""):].replace("\"},\"callToAction\":{\"","")[:4]
        print(game_year)
        game_name = game_name[:game_name.find("},\"subtitle\":{")-1]
        # print(game_name)
        # if( "\":{\"clickTrackingParams\":\"" or "\":\"" in game_year):
        #     game_year="N/A"
        # if("igationEndpoint\":{\"clickTrackingParams\":\"" or "\":\"" in game_name):
        #     game_name="N/A"


        likes_indx = str(resp.content).find("{\"iconType\":\"LIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"") + len("{\"iconType\":\"LIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"")
        likes = str(resp.content)[likes_indx:likes_indx + 50]
        likes = str(likes[:likes.find("likes")])
        likes = likes.strip()
        likes= likes.replace(",", "")
        likes = int(likes)

        dislikes_indx = str(resp.content).find("\"defaultIcon\":{\"iconType\":\"DISLIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"") + len("\"defaultIcon\":{\"iconType\":\"DISLIKE\"},\"defaultText\":{\"accessibility\":{\"accessibilityData\":{\"label\":\"")
        dislikes = str(resp.content)[dislikes_indx:dislikes_indx + 50]
        dislikes = str(dislikes[:dislikes.find("dislikes")])
        dislikes = dislikes.strip()
        dislikes= dislikes.replace(",", "")
        dislikes = int(dislikes)
        

        viewcount_indx = str(resp.content).find("\"viewCount\":{\"videoViewCountRenderer\":{\"viewCount\":{\"simpleText\":\"") + len("\"viewCount\":{\"videoViewCountRenderer\":{\"viewCount\":{\"simpleText\":\"")
        viewcount = str(resp.content)[viewcount_indx:viewcount_indx + 50]
        viewcount = str(viewcount[:viewcount.find(" views")])
        viewcount = viewcount.strip()
        viewcount= viewcount.replace(",", "")
        viewcount = int(viewcount)

        # comment_count_indx = str(resp.content).find() + len()
        # comment_count = str(resp.content)[comment_count_indx:comment_count_indx + 50]
        # comment_count = str(comment_count[:comment_count.find(" comment_s")])
        # comment_count = comment_count.strip()
        # comment_count= comment_count.replace(",", "")
        # comment_count = int(comment_count)

        print("Game Year of Pub: ", game_year)
        print("Game Name: ",        game_name)
        print("Likes: ",            likes)
        print("ViewCount",          viewcount)
        print("Dislikes: ",         dislikes)
        print("ViewCount / SubscriberCount: ",         viewcount / self.subscriberCount)
        print("SubscriberCount: ",self.subscriberCount)
        
        output = {
        "game_year": game_year,
        "game_name":        game_name,
        "likes":        likes,
        "views":          viewcount,
        "dislikes":         dislikes,
        "l/d":         likes/dislikes
        }

        # videos = data["items"]
        # nextPageToken = data["nextPageToken"]

    def getChannelVids(self):
        url = "https://www.googleapis.com/youtube/v3/search?key=" + str(self.api_key) + "&channelId=" + str(self.channelID) + "&part=snippet,id&order=date&maxResults=" + str(self.targetAmount)
        print(url)
        resp = requests.get(url)
        self.total_requests+=1
        data = resp.content
        data = json.loads(data)
        videos = data["items"]
        self.nextPageToken = data["nextPageToken"]
    # CHECK LIVE CONTENT = NONE
    #  "liveBroadcastContent": "none",
        cnt = 0
        for j in range(self.targetAmount//50):
            for count,i in enumerate(videos):
                if(count%15==0):
                    time.sleep(1)   
                print(count + (j) * 50, " - ", i["snippet"]["title"])
                self.counter+=1
            # if(j == self.targetAmount//50 - 1):
            #     print("END OF LINE")
            try:
                url = "https://www.googleapis.com/youtube/v3/search?key=" + str(self.api_key) + "&channelId=" + str(self.channelID) + "&part=snippet,id&order=date&maxResults=" + str(self.targetAmount) + "&pageToken=" + self.nextPageToken
                print(url)
                resp = requests.get(url)
                self.total_requests+=1
                data = resp.content
                data = json.loads(data)
                videos = data["items"]
                self.nextPageToken = data["nextPageToken"]
            except Exception as e:
                print(e)
            
        
                
                    
            
dl = channelDownloader(api_key = Api_key, channelID="UCxSM0Qt-QVywFjYo0wLnd0Q",targetAmount=100)
# dl.getChannelVids()
# dl.getVidInfo("B0N-qYLP2_4")






