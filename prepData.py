from pyyoutube import Api
key = open("apikey.txt", 'r')
key = str(key.read())


# api = Api(api_key=key)


api = Api(client_id="", client_secret="")
URL, Thing = api.get_authorization_url()
print(URL)


api.generate_access_token(authorization_response=URL)