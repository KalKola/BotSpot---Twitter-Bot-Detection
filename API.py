import json
import csv
import tweepy
import pandas as pd

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'
hashtag = '#maga'

def search(ck, ck_secret, at, at_secret, hashtag):

    #using access tokens and consumer keys to set up the API
    auth = tweepy.OAuthHandler(ck, ck_secret)
    auth.set_access_token(at, at_secret)
    api = tweepy.API(auth)

    #create the csv file to write the data to
    fname = 'dataset'
    with open('%s.csv' % (fname), 'w') as file:

            fileWriter = csv.writer(file)

            fileWriter.writerow(['name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count',
                                 'favourites_count', 'listed_count', 'time_zone', 'location', 'default_profile',
                                 'default_profile_image', 'geo_enabled', 'profile_text_color', 'profile_background_tile',
                                 'profile_background_color', 'profile_link_color', 'protected', 'verified', 'notifications',
                                 'description', 'contributors_enabled', 'following', 'created_at'])


            for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", tweet_mode='extended').items(1000):

                fileWriter.writerow([
                            tweet.user.name.encode('utf-8'),
                            tweet.user.screen_name.encode('utf-8'),

                            tweet.user.statuses_count,
                            tweet.user.followers_count,
                            tweet.user.friends_count,
                            tweet.user.favourites_count,
                            tweet.user.listed_count,

                            tweet.user.time_zone,
                            tweet.user.location.encode('utf-8'),
                            tweet.user.default_profile,
                            tweet.user.default_profile_image,
                            tweet.user.geo_enabled,
                            tweet.user.profile_text_color,
                            tweet.user.profile_background_tile,
                            tweet.user.profile_background_color,
                            tweet.user.profile_link_color,

                            tweet.user.protected,
                            tweet.user.verified,
                            tweet.user.notifications,

                            tweet.user.description.encode('utf-8'),
                            tweet.user.contributors_enabled,
                            tweet.user.following,
                            tweet.user.created_at
                            ])


#convert profile link color to Boolean for 1DA1F2 (blue)
search(ck, ck_secret, at, at_secret, hashtag)