import json
import csv
import tweepy

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'
hashtag = '#lol'

def search(ck, ck_secret, at, at_secret, hashtag):

    #using access tokens and consumer keys to set up the API
    auth = tweepy.OAuthHandler(ck, ck_secret)
    auth.set_access_token(at, at_secret)
    api = tweepy.API(auth)

    #create the csv file to write the data to
    fname = 'dataset'
    with open('%s.csv' % (fname), 'w') as file:

            fileWriter = csv.writer(file)

            fileWriter.writerow(['Screen Name', 'Name', 'Follower Count', 'Retweet Count',
                                 'Favorite Count', 'Statuses Count', 'Friends Count',
                                 'Favorites Count', 'Listed Count', 'Tweet Text',
                                 'User Creation', 'Hashtags', 'Location', 'Time Zone',
                                 'Profile Image', 'Description', 'Profile Link Color',
                                 'Profile Background Color', 'Default Profile Image'])


            for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", tweet_mode='extended').items(1000):

                fileWriter.writerow([
                            tweet.user.screen_name.encode('utf-8'),
                            tweet.user.name.encode('utf-8'),

                            tweet.user.followers_count,
                            tweet.retweet_count,
                            tweet.favorite_count,
                            tweet.user.statuses_count,
                            tweet.user.friends_count,
                            tweet.user.favourites_count,
                            tweet.user.listed_count,

                            tweet.full_text.replace('\n', ' ').encode('utf-8'),
                            tweet.user.created_at,
                            [hashtags['text'].encode('utf-8') for hashtags in tweet._json['entities']['hashtags']],
                            tweet.user.location.encode('utf-8'),
                            tweet.user.time_zone,
                            tweet.user.profile_image_url,
                            tweet.user.description.encode('utf-8'),
                            tweet.user.profile_link_color,
                            tweet.user.profile_background_color,
                            tweet.user.default_profile_image])

#Converting Default Profile Image to Boolean
#DefaultNew = df.housing.map(dict(yes=1, no=0))

#convert profile link color to Boolean for 1DA1F2 (blue)
search(ck, ck_secret, at, at_secret, hashtag)