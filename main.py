import json
import csv
import tweepy

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'
hashtag = '#aquaman'

def search(ck, ck_secret, at, at_secret, hashtag):

    #using access tokens and consumer keys to set up the API
    auth = tweepy.OAuthHandler(ck, ck_secret)
    auth.set_access_token(at, at_secret)
    api = tweepy.API(auth)

    #create the csv file to write the data to
    fname = 'dataset'
    with open('%s.csv' % (fname), 'w') as file:

            fileWriter = csv.writer(file)

            fileWriter.writerow(['Username', 'Text', 'Timestamp', 'Hashtags', 'Follower no.',
                                 'User Location', 'Retweet no.', 'Favorite no.', 'Profile Image',
                                 'Profile Description', 'Profile Link Color', 'Profile Background Color',
                                 'Name', 'Statuses Count', 'Friends Count', 'Favorites Count', 'Listed Count',
                                 'Time Zone', 'Default Profile', 'Profile Background Color'])


            for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", tweet_mode='extended').items(100):

                fileWriter.writerow([
                            tweet.user.screen_name.encode('utf-8'),
                            tweet.full_text.replace('\n', ' ').encode('utf-8'),
                            tweet.user.created_at,
                            [hashtags['text'].encode('utf-8') for hashtags in tweet._json['entities']['hashtags']],
                            tweet.user.followers_count,
                            tweet.user.location.encode('utf-8'),
                            tweet.retweet_count,
                            tweet.favorite_count,
                            tweet.user.profile_image_url,
                            tweet.user.description.encode('utf-8'),
                            tweet.user.profile_link_color,
                            tweet.user.profile_background_color,
                            tweet.user.name.encode('utf-8'),
                            tweet.user.statuses_count,
                            tweet.user.friends_count,
                            tweet.user.favourites_count,
                            tweet.user.listed_count,
                            tweet.user.time_zone,
                            tweet.user.default_profile_image,
                            tweet.user.profile_background_color])

#convert profile link color to Boolean for 1DA1F2 (blue)
search(ck, ck_secret, at, at_secret, hashtag)