# crisis_detector.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz
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
        
        # Clean text - handle both 'text' and 'title' columns
        text_column = 'text' if 'text' in df.columns else 'title'
        df['cleaned_text'] = df[text_column].fillna('').apply(self.clean_text)
        
        # If both text and title exist, combine them
        if 'text' in df.columns and 'title' in df.columns:
            df['cleaned_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(self.clean_text)
        
        # Analyze sentiment
        sentiment_results = df['cleaned_text'].apply(self.analyze_sentiment)
        sentiment_df = pd.json_normalize(sentiment_results)
        
        # Detect crisis keywords
        crisis_results = df['cleaned_text'].apply(self.detect_crisis_keywords)
        crisis_df = pd.json_normalize(crisis_results)
        
        # Combine all results
        processed_df = pd.concat([df, sentiment_df, crisis_df], axis=1)
        
        # Calculate engagement-weighted sentiment using 'score' column
        processed_df['engagement_score'] = processed_df['score'].fillna(0)
        processed_df['weighted_sentiment'] = processed_df['final_score'] * np.log1p(abs(processed_df['engagement_score']))
        
        return processed_df
    
    def detect_crisis(self, df, lookback_hours=24):
        """Detect potential crisis situations"""
        # Initialize default return values
        default_result = {
            'crisis_level': 'low',
            'crisis_score': 0,
            'total_posts': 0,
            'negative_ratio': 0,
            'avg_sentiment': 0,
            'alerts': [],
            'recommendations': ['Continue regular monitoring'],
            'top_negative_posts': []
        }
        
        if df.empty:
            default_result['alerts'] = ['No data available']
            default_result['recommendations'] = ['Increase data collection']
            return default_result
        
        # Handle timezone-aware datetime comparison
        try:
            # Convert created_date to datetime if it's not already
            df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['created_date'])
            
            if df.empty:
                default_result['alerts'] = ['No valid date data available']
                return default_result
            
            # Create cutoff_time - make it timezone-aware if the data is timezone-aware
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Check if the datetime column is timezone-aware
            if df['created_date'].dt.tz is not None:
                # If data is timezone-aware, make cutoff_time timezone-aware too
                if cutoff_time.tzinfo is None:
                    cutoff_time = pytz.UTC.localize(cutoff_time)
            else:
                # If data is timezone-naive, ensure cutoff_time is also naive
                if cutoff_time.tzinfo is not None:
                    cutoff_time = cutoff_time.replace(tzinfo=None)
            
            # Filter recent data
            recent_df = df[df['created_date'] >= cutoff_time].copy()
            
            # If no recent data, use all data but add a warning
            if recent_df.empty:
                recent_df = df.copy()
                default_result['alerts'].append(f'No data from last {lookback_hours} hours, analyzing all available data')
            
        except Exception as e:
            print(f"Warning: Error in datetime filtering: {e}")
            # Fallback: use all data if datetime filtering fails
            recent_df = df.copy()
            default_result['alerts'].append('Date filtering failed, analyzing all available data')
        
        # Ensure we have required columns
        required_columns = ['final_sentiment', 'final_score', 'crisis_score', 'engagement_score']
        for col in required_columns:
            if col not in recent_df.columns:
                print(f"Warning: Missing column {col}")
                default_result['alerts'].append(f'Missing required data: {col}')
                return default_result
        
        # Calculate crisis indicators
        total_posts = len(recent_df)
        negative_posts = len(recent_df[recent_df['final_sentiment'] == 'negative'])
        negative_ratio = negative_posts / total_posts if total_posts > 0 else 0
        
        avg_sentiment = recent_df['final_score'].mean()
        crisis_keyword_posts = len(recent_df[recent_df['crisis_score'] > 0])
        
        # Calculate high engagement threshold safely
        engagement_threshold = recent_df['engagement_score'].quantile(0.75) if len(recent_df) > 0 else 0
        high_engagement_negative = len(recent_df[
            (recent_df['final_sentiment'] == 'negative') & 
            (recent_df['engagement_score'] > engagement_threshold)
        ])
        
        # Calculate overall crisis score
        crisis_score = 0
        if total_posts > 0:
            crisis_score = (
                (negative_ratio * 40) +  # 40% weight to negative ratio
                (abs(avg_sentiment) * 30 if avg_sentiment < 0 else 0) +  # 30% weight to sentiment
                ((crisis_keyword_posts / total_posts) * 20) +  # 20% weight to crisis keywords
                ((high_engagement_negative / total_posts) * 10)  # 10% weight to viral negative posts
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
        
        # Get top negative posts safely
        top_negative_posts = []
        try:
            negative_posts_df = recent_df[recent_df['final_sentiment'] == 'negative']
            if not negative_posts_df.empty:
                columns_to_include = []
                if 'text' in negative_posts_df.columns:
                    columns_to_include.append('text')
                elif 'title' in negative_posts_df.columns:
                    columns_to_include.append('title')
                
                if 'platform' in negative_posts_df.columns:
                    columns_to_include.append('platform')
                
                columns_to_include.append('engagement_score')
                
                top_negative_posts = negative_posts_df.nlargest(5, 'engagement_score')[columns_to_include].to_dict('records')
        except Exception as e:
            print(f"Warning: Error getting top negative posts: {e}")
        
        return {
            'crisis_level': crisis_level,
            'crisis_score': round(crisis_score, 2),
            'total_posts': total_posts,
            'negative_ratio': round(negative_ratio, 3),
            'avg_sentiment': round(avg_sentiment, 3),
            'alerts': alerts,
            'recommendations': recommendations,
            'top_negative_posts': top_negative_posts
        }
    
    def generate_insights(self, df):
        """Generate insights and visualizations"""
        if df.empty:
            return "No data available for analysis"
        
        insights = {
            'summary': {
                'total_posts': len(df),
                'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
                'sentiment_distribution': df['final_sentiment'].value_counts().to_dict() if 'final_sentiment' in df.columns else {},
                'avg_sentiment': df['final_score'].mean() if 'final_score' in df.columns else 0,
                'crisis_posts': len(df[df['crisis_score'] > 0]) if 'crisis_score' in df.columns else 0
            },
            'trends': self._analyze_trends(df),
            'top_keywords': self._extract_keywords(df)
        }
        
        return insights
    
    def _analyze_trends(self, df):
        """Analyze sentiment trends over time"""
        try:
            if 'created_date' not in df.columns or 'final_score' not in df.columns:
                return {'daily_sentiment': {}, 'trend_direction': 'unknown'}
                
            df['date'] = pd.to_datetime(df['created_date'], errors='coerce').dt.date
            df = df.dropna(subset=['date'])
            
            if df.empty:
                return {'daily_sentiment': {}, 'trend_direction': 'unknown'}
                
            daily_sentiment = df.groupby('date')['final_score'].mean()
            
            return {
                'daily_sentiment': daily_sentiment.to_dict(),
                'trend_direction': 'improving' if len(daily_sentiment) > 1 and daily_sentiment.iloc[-1] > daily_sentiment.iloc[0] else 'declining'
            }
        except Exception as e:
            print(f"Warning: Error in trend analysis: {e}")
            return {'daily_sentiment': {}, 'trend_direction': 'unknown'}
    
    def _extract_keywords(self, df):
        """Extract most common words from negative posts"""
        try:
            if 'final_sentiment' not in df.columns or 'cleaned_text' not in df.columns:
                return []
                
            negative_texts = df[df['final_sentiment'] == 'negative']['cleaned_text']
            if negative_texts.empty:
                return []
            
            all_text = ' '.join(negative_texts.fillna(''))
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Remove common stop words
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'])
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            from collections import Counter
            word_freq = Counter(filtered_words)
            
            return word_freq.most_common(10)
        except Exception as e:
            print(f"Warning: Error in keyword extraction: {e}")
            return []

# Example usage
if __name__ == "__main__":
    detector = CrisisDetector()
    
    # Load your data
    try:
        # Try to load any available data file
        import os
        data_files = []
        if os.path.exists('data'):
            data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        
        if not data_files:
            print("No CSV data files found in 'data' directory.")
            print("Please ensure you have a CSV file with social media data.")
            exit()
        
        # Use the first available data file
        data_file = os.path.join('data', data_files[0])
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} records from {data_file}")
        
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
        if crisis_report['alerts']:
            for alert in crisis_report['alerts']:
                print(f"⚠️  {alert}")
        else:
            print("✅ No alerts")
        
        print("\nRecommendations:")
        for rec in crisis_report['recommendations']:
            print(f"💡 {rec}")
        
        # Show top negative posts if any
        if crisis_report['top_negative_posts']:
            print(f"\nTop {len(crisis_report['top_negative_posts'])} Most Engaging Negative Posts:")
            for i, post in enumerate(crisis_report['top_negative_posts'], 1):
                text_key = 'text' if 'text' in post else 'title'
                text_preview = post.get(text_key, 'No text')[:100] + "..." if len(str(post.get(text_key, ''))) > 100 else post.get(text_key, 'No text')
                print(f"{i}. Score: {post.get('engagement_score', 0)} | {text_preview}")
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        processed_df.to_csv('data/processed_data.csv', index=False)
        print(f"\nProcessed data saved to 'data/processed_data.csv'")
        
        # Generate additional insights
        insights = detector.generate_insights(processed_df)
        print(f"\n=== ADDITIONAL INSIGHTS ===")
        print(f"Total posts analyzed: {insights['summary']['total_posts']}")
        print(f"Platform distribution: {insights['summary']['platforms']}")
        print(f"Sentiment distribution: {insights['summary']['sentiment_distribution']}")
        print(f"Average sentiment: {insights['summary']['avg_sentiment']:.3f}")
        print(f"Posts with crisis keywords: {insights['summary']['crisis_posts']}")
        
        if insights['top_keywords']:
            print(f"\nTop keywords in negative posts:")
            for word, count in insights['top_keywords'][:5]:
                print(f"  {word}: {count}")
        
    except FileNotFoundError:
        print("Data file not found. Please ensure you have a CSV file with social media data in the 'data' directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your data file format and try again.")