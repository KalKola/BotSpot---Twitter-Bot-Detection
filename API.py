import csv
import tweepy


def search(c_key, c_key_secret, acc_token, acc_token_secret):

    print("Enter Hashtag Below: ")
    set_hashtag = input()

    print("Connecting to Twitter API")
    # initialize oAuth verification using consumer-key & access-token
    auth_init = tweepy.OAuthHandler(c_key, c_key_secret)
    auth_init.set_access_token(acc_token, acc_token_secret)
    # establish connection to Twitter API - set auto-wait on call-limit
    t_api = tweepy.API(auth_init, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # create the csv file to write the data to
    fname = 'dataset'
    with open('%s.csv' % fname, 'w') as file:

        file_writer = csv.writer(file)

        file_writer.writerow(['name', 'screen_name', 'statuses_count', 'followers_count',
                              'friends_count', 'favourites_count', 'listed_count', 'time_zone',
                              'location', 'default_profile', 'default_profile_image', 'geo_enabled',
                              'profile_text_color', 'profile_background_tile', 'profile_background_color',
                              'profile_link_color', 'protected', 'verified', 'notifications',
                              'description', 'contributors_enabled', 'following', 'created_at'])

        print("Fetching Tweets from " + set_hashtag)

        for tweet in tweepy.Cursor(t_api.search, q=set_hashtag, lang="en", tweet_mode='extended').items(1000):

            file_writer.writerow([
                        tweet.user.name.encode('utf-8'),
                        tweet.user.screen_name.encode('utf-8'),

                        tweet.user.statuses_count,
                        tweet.user.followers_count,
                        tweet.user.friends_count,
                        tweet.user.favourites_count,
                        tweet.user.listed_count,

                        tweet.user.time_zone,
                        tweet.user.location.encode('utf-8'),
                        int(tweet.user.default_profile),
                        int(tweet.user.default_profile_image),
                        int(tweet.user.geo_enabled),
                        tweet.user.profile_text_color,
                        int(tweet.user.profile_background_tile),
                        tweet.user.profile_background_color,
                        tweet.user.profile_link_color,

                        int(tweet.user.protected),
                        int(tweet.user.verified),
                        int(tweet.user.notifications),

                        tweet.user.description.encode('utf-8'),
                        int(tweet.user.contributors_enabled),
                        tweet.user.following,
                        tweet.user.created_at
                        ])
    return


