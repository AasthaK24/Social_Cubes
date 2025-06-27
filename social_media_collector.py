import praw
import pandas as pd
import tweepy
from datetime import datetime, timedelta
import json
import time

class SocialMediaCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id="mNr2FHzEIaZBzgMpvcKaZg",
            client_secret="_x8SZb1mjmOfUPiC3MbJtOfyWFsrtg",
            user_agent= "MyApp"
        )

        self.twitter_api = None
    
    def setup_twitter(self, bearer_token):
        self.twitter_api = tweepy.Client(bearer_token=bearer_token)

    def collect_reddit_data(self, business_name, subreddits=['all'], limit=100):
        posts_data = []

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for posts mentioning the business
                for post in subreddit.search(business_name, limit=limit, time_filter='week'):
                    posts_data.append({
                        'platform': 'reddit',
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_date': datetime.fromtimestamp(post.created_utc),
                        'subreddit': post.subreddit.display_name,
                        'url': f"https://reddit.com{post.permalink}",
                        'author': str(post.author) if post.author else 'deleted'
                    })

                    # Get top comments
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:5]: #Top 5 comments
                        posts_data.append({
                            'platform': 'reddit',
                            'id': comment.id,
                            'title': f"Comment on: {post.title}",
                            'text': comment.body,
                            'score': comment.score,
                            'created_date': datetime.fromtimestamp(comment.created_utc),
                            'subreddit': post.subreddit.display_name,
                            'url': f"https://reddit.com{comment.permalink}",
                            'author': str(comment.author) if comment.author else 'deleted'
                        })

                time.sleep(1)  # Be respectful to API limits
                
            except Exception as e:
                print(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        return pd.DataFrame(posts_data)
    
    def collect_twitter_data(self, business_name, max_results=100):
        if not self.twitter_api:
            print("Twitter API not configured")
            return pd.DataFrame()
        
        try:
            tweets = tweepy.Paginator(
                self.twitter_api.search_recent_tweets,
                query=f"{business_name} -is:retweet",
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
            ).flatten(limit=max_results)

            tweets_data=[]
            for tweet in tweets:
                tweets_data.append({
                    'platform': 'twitter',
                    'id': tweet.id,
                    'title': tweet.text[:50] + "..." if len(tweet.text) > 50 else tweet.text,
                    'text': tweet.text,
                    'score': tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count'],
                    'created_date':tweet.created_at,
                    'author': tweet.author_id,
                    'url': f"https://twitter.com/i/status/{tweet.id}"
                })
            return pd.DataFrame(tweets_data)
        
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
            return pd.DataFrame()
        
    def save_data(self, df, filename):
        """Save collected data"""
        df.to_csv(f"data/{filename}", index=False)
        print(f"Saved {len(df)} records to {filename}")

# Example usage
if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)
    
    collector = SocialMediaCollector()

    collector.setup_twitter("AAAAAAAAAAAAAAAAAAAAAIOp2gEAAAAAb3qtd5tsMCI9uBUrMDNTSypPApA%3DeHsjxSeVWwF7jyrAU5COnHDlPiVec8Kwyd1feSumAOqAPc45s3")
    # Test with a well-known brand
    business_name = "Nintendo"  # Change this to test different businesses
    
    print("Collecting Reddit data...")
    reddit_data = collector.collect_reddit_data(
        business_name, 
        subreddits=['all', 'console', 'gaming'], 
        limit=50
    )
    
    if not reddit_data.empty:
        collector.save_data(reddit_data, f"{business_name.lower()}_reddit_data.csv")
        print(f"Collected {len(reddit_data)} Reddit posts/comments")
    else:
        print("No Reddit data collected")
    
    print("Collecting Twitter data...")
    twitter_data = collector.collect_twitter_data(business_name, max_results=50)

    if not twitter_data.empty:
        collector.save_data(twitter_data, f"{business_name.lower()}_twitter_data.csv")
        print(f"Collected {len(twitter_data)} Twitter posts")
    else:
        print("No Twitter data collected")
