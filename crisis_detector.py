# crisis_detector.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class CrisisDetector:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.crisis_keywords = [
            'boycott', 'scandal', 'lawsuit', 'illegal', 'scam', 'fraud',
            'terrible', 'worst', 'disgusting', 'horrible', 'awful',
            'discrimination', 'racist', 'sexist', 'harassment',
            'food poisoning', 'sick', 'contaminated', 'expired',
            'rude', 'unprofessional', 'manager', 'complaint'
        ]
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple methods"""
        if not text:
            return {'compound': 0, 'textblob': 0, 'final_sentiment': 'neutral'}
        
        # VADER sentiment (good for social media)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combined sentiment score
        final_score = (vader_scores['compound'] + textblob_polarity) / 2
        
        # Classify sentiment
        if final_score <= -0.1:
            sentiment = 'negative'
        elif final_score >= 0.1:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
            
        return {
            'compound': vader_scores['compound'],
            'textblob': textblob_polarity,
            'final_score': final_score,
            'final_sentiment': sentiment,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
    
    def detect_crisis_keywords(self, text):
        """Detect crisis-related keywords"""
        if not text:
            return {'crisis_score': 0, 'found_keywords': []}
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in self.crisis_keywords if keyword in text_lower]
        crisis_score = len(found_keywords)
        
        return {
            'crisis_score': crisis_score,
            'found_keywords': found_keywords
        }
    
    def process_data(self, df):
        """Process the collected social media data"""
        if df.empty:
            return df
        
        print("Processing social media data...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Analyze sentiment
        sentiment_results = df['cleaned_text'].apply(self.analyze_sentiment)
        sentiment_df = pd.json_normalize(sentiment_results)
        
        # Detect crisis keywords
        crisis_results = df['cleaned_text'].apply(self.detect_crisis_keywords)
        crisis_df = pd.json_normalize(crisis_results)
        
        # Combine all results
        processed_df = pd.concat([df, sentiment_df, crisis_df], axis=1)
        
        # Calculate engagement-weighted sentiment
        processed_df['engagement_score'] = processed_df['score'].fillna(0)
        processed_df['weighted_sentiment'] = processed_df['final_score'] * np.log1p(processed_df['engagement_score'])
        
        return processed_df
    
    def detect_crisis(self, df, lookback_hours=24):
        """Detect potential crisis situations"""
        if df.empty:
            return {
                'crisis_level': 'low',
                'crisis_score': 0,
                'alerts': [],
                'recommendations': []
            }
        
        # Filter recent data
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_df = df[pd.to_datetime(df['created_date']) >= cutoff_time].copy()
        
        if recent_df.empty:
            return {
                'crisis_level': 'low',
                'crisis_score': 0,
                'alerts': ['No recent data available'],
                'recommendations': ['Increase monitoring frequency']
            }
        
        # Calculate crisis indicators
        total_posts = len(recent_df)
        negative_posts = len(recent_df[recent_df['final_sentiment'] == 'negative'])
        negative_ratio = negative_posts / total_posts if total_posts > 0 else 0
        
        avg_sentiment = recent_df['final_score'].mean()
        crisis_keyword_posts = len(recent_df[recent_df['crisis_score'] > 0])
        
        high_engagement_negative = len(recent_df[
            (recent_df['final_sentiment'] == 'negative') & 
            (recent_df['engagement_score'] > recent_df['engagement_score'].quantile(0.75))
        ])
        
        # Calculate overall crisis score
        crisis_score = (
            (negative_ratio * 40) +  # 40% weight to negative ratio
            (abs(avg_sentiment) * 30 if avg_sentiment < 0 else 0) +  # 30% weight to sentiment
            (crisis_keyword_posts / total_posts * 20) +  # 20% weight to crisis keywords
            (high_engagement_negative / total_posts * 10)  # 10% weight to viral negative posts
        )
        
        # Determine crisis level
        if crisis_score >= 60:
            crisis_level = 'high'
        elif crisis_score >= 30:
            crisis_level = 'medium'
        else:
            crisis_level = 'low'
        
        # Generate alerts and recommendations
        alerts = []
        recommendations = []
        
        if negative_ratio > 0.6:
            alerts.append(f"High negative sentiment: {negative_ratio:.1%} of recent posts are negative")
            
        if crisis_keyword_posts > 0:
            alerts.append(f"Crisis keywords detected in {crisis_keyword_posts} posts")
            
        if high_engagement_negative > 0:
            alerts.append(f"{high_engagement_negative} high-engagement negative posts detected")
        
        # Generate recommendations based on crisis level
        if crisis_level == 'high':
            recommendations.extend([
                "Immediate response required - prepare official statement",
                "Monitor social media continuously",
                "Consider proactive customer outreach",
                "Escalate to crisis management team"
            ])
        elif crisis_level == 'medium':
            recommendations.extend([
                "Increased monitoring recommended",
                "Prepare response templates",
                "Engage with concerned customers",
                "Monitor for escalation"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain positive engagement",
                "Document patterns for future reference"
            ])
        
        return {
            'crisis_level': crisis_level,
            'crisis_score': round(crisis_score, 2),
            'total_posts': total_posts,
            'negative_ratio': round(negative_ratio, 3),
            'avg_sentiment': round(avg_sentiment, 3),
            'alerts': alerts,
            'recommendations': recommendations,
            'top_negative_posts': recent_df[recent_df['final_sentiment'] == 'negative'].nlargest(5, 'engagement_score')[['text', 'platform', 'engagement_score']].to_dict('records')
        }
    
    def generate_insights(self, df):
        """Generate insights and visualizations"""
        if df.empty:
            return "No data available for analysis"
        
        insights = {
            'summary': {
                'total_posts': len(df),
                'platforms': df['platform'].value_counts().to_dict(),
                'sentiment_distribution': df['final_sentiment'].value_counts().to_dict(),
                'avg_sentiment': df['final_score'].mean(),
                'crisis_posts': len(df[df['crisis_score'] > 0])
            },
            'trends': self._analyze_trends(df),
            'top_keywords': self._extract_keywords(df)
        }
        
        return insights
    
    def _analyze_trends(self, df):
        """Analyze sentiment trends over time"""
        df['date'] = pd.to_datetime(df['created_date']).dt.date
        daily_sentiment = df.groupby('date')['final_score'].mean()
        
        return {
            'daily_sentiment': daily_sentiment.to_dict(),
            'trend_direction': 'improving' if daily_sentiment.iloc[-1] > daily_sentiment.iloc[0] else 'declining'
        }
    
    def _extract_keywords(self, df):
        """Extract most common words from negative posts"""
        negative_texts = df[df['final_sentiment'] == 'negative']['cleaned_text']
        if negative_texts.empty:
            return []
        
        all_text = ' '.join(negative_texts)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Remove common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'])
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        from collections import Counter
        word_freq = Counter(filtered_words)
        
        return word_freq.most_common(10)

# Example usage
if __name__ == "__main__":
    detector = CrisisDetector()
    
    # Load your data
    try:
        df = pd.read_csv('data/starbucks_twitter_data.csv')  # Change filename as needed
        print(f"Loaded {len(df)} records")
        
        # Process the data
        processed_df = detector.process_data(df)
        
        # Detect crisis
        crisis_report = detector.detect_crisis(processed_df)
        
        print("\n=== CRISIS DETECTION REPORT ===")
        print(f"Crisis Level: {crisis_report['crisis_level'].upper()}")
        print(f"Crisis Score: {crisis_report['crisis_score']}/100")
        print(f"Total Recent Posts: {crisis_report['total_posts']}")
        print(f"Negative Ratio: {crisis_report['negative_ratio']:.1%}")
        
        print("\nAlerts:")
        for alert in crisis_report['alerts']:
            print(f"⚠️  {alert}")
        
        print("\nRecommendations:")
        for rec in crisis_report['recommendations']:
            print(f"💡 {rec}")
        
        # Save processed data
        processed_df.to_csv('data/processed_data.csv', index=False)
        print("\nProcessed data saved to 'data/processed_data.csv'")
        
    except FileNotFoundError:
        print("No data file found. Run the social_media_collector.py first!")