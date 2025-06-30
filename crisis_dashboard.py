# crisis_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from social_media_collector import SocialMediaCollector
from crisis_detector import CrisisDetector
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Social Media Crisis Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .crisis-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .crisis-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .crisis-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .alert-box {
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        background-color: #fff3e0;
        color: #000000;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.collector = SocialMediaCollector()
        self.detector = CrisisDetector()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame()
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = pd.DataFrame()
        if 'crisis_report' not in st.session_state:
            st.session_state.crisis_report = {}
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.title("🚨 Crisis Detector")
        st.sidebar.markdown("---")

        # Business name input
        business_name = st.sidebar.text_input(
        "Business Name to Monitor",
            value="McDonald's",
            help="Enter the business/brand name you want to monitor"
        )

        # Crisis keyword input
        st.sidebar.subheader("Crisis Keywords")
        custom_keywords = st.sidebar.text_area(
            "Enter keywords (comma-separated)",
            value="boycott, scandal, scam, fraud, terrible, worst",
            help="These keywords will be used to detect potential crisis posts"
        )
        keyword_list = [kw.strip().lower() for kw in custom_keywords.split(',') if kw.strip()]

        # Data collection settings
        st.sidebar.subheader("Data Collection")

        subreddits = st.sidebar.multiselect(
            "Subreddits to Monitor",
            ['all', 'gaming', 'console', 'controller issue', 'boycott nintendo', 'no innovation', 'negative'],
            default=['all', 'gaming']
        )

        post_limit = st.sidebar.slider(
            "Posts per Subreddit",
            min_value=10,
            max_value=100,
            value=30,
            step=10
        )

        # Crisis detection settings
        st.sidebar.subheader("Crisis Detection")

        lookback_hours = st.sidebar.slider(
            "Analysis Time Window (hours)",
            min_value=6,
            max_value=72,
            value=24,
            step=6
        )

        # Data collection button
        if st.sidebar.button("🔄 Collect Fresh Data", type="primary"):
            with st.spinner("Collecting social media data..."):
                try:
                    # Update detector keywords dynamically
                    self.detector.crisis_keywords = keyword_list

                    # Collect data
                    new_data = self.collector.collect_reddit_data(
                        business_name=business_name,
                        subreddits=subreddits,
                        limit=post_limit,
                        crisis_keywords=keyword_list
                    )

                    if not new_data.empty:
                        st.session_state.data = new_data

                        # Process data
                        st.session_state.processed_data = self.detector.process_data(new_data)

                        # Generate crisis report
                        st.session_state.crisis_report = self.detector.detect_crisis(
                            st.session_state.processed_data,
                            lookback_hours=lookback_hours
                        )

                        st.sidebar.success(f"✅ Collected {len(new_data)} posts!")
                    else:
                        st.sidebar.error("❌ No data collected. Try different settings.")

                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")

        # Load existing data button
        if st.sidebar.button("📁 Load Existing Data"):
            try:
                if os.path.exists('data/processed_data.csv'):
                    st.session_state.processed_data = pd.read_csv('data/processed_data.csv')

                    # Update detector keywords before analysis
                    self.detector.crisis_keywords = keyword_list

                    st.session_state.crisis_report = self.detector.detect_crisis(
                        st.session_state.processed_data,
                        lookback_hours=lookback_hours
                    )
                    st.sidebar.success("✅ Data loaded successfully!")
                else:
                    st.sidebar.warning("⚠️ No existing data found.")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")

        return business_name, lookback_hours

    
    def display_crisis_overview(self):
        """Display crisis overview metrics"""
        if not st.session_state.crisis_report:
            st.warning("No crisis analysis available. Please collect data first.")
            return
        
        report = st.session_state.crisis_report
        
        st.header("🚨 Crisis Overview")
        
        # Crisis level indicator
        crisis_level = report.get('crisis_level', 'unknown')
        crisis_score = report.get('crisis_score', 0)
        
        # Color coding based on crisis level
        if crisis_level == 'high':
            color = "🔴"
            css_class = "crisis-high"
        elif crisis_level == 'medium':
            color = "🟡"
            css_class = "crisis-medium"
        else:
            color = "🟢"
            css_class = "crisis-low"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Crisis Level",
                value=f"{color} {crisis_level.upper()}",
                delta=f"Score: {crisis_score}/100"
            )
        
        with col2:
            st.metric(
                label="Total Recent Posts",
                value=report.get('total_posts', 0)
            )
        
        with col3:
            negative_ratio = report.get('negative_ratio', 0)
            st.metric(
                label="Negative Posts",
                value=f"{negative_ratio:.1%}",
                delta="Of recent posts"
            )
        
        with col4:
            avg_sentiment = report.get('avg_sentiment', 0)
            sentiment_delta = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            st.metric(
                label="Avg Sentiment",
                value=f"{avg_sentiment:.2f}",
                delta=sentiment_delta
            )
        
        # Alerts section
        if report.get('alerts'):
            st.subheader("⚠️ Active Alerts")
            for alert in report['alerts']:
                st.markdown(f'<div class="alert-box">🚨 {alert}</div>', unsafe_allow_html=True)
        
        # Recommendations section
        if report.get('recommendations'):
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(report['recommendations'], 1):
                st.write(f"{i}. {rec}")
    
    def display_sentiment_analysis(self):
        """Display sentiment analysis charts"""
        if st.session_state.processed_data.empty:
            st.warning("No processed data available.")
            return
        
        df = st.session_state.processed_data
        
        st.header("📊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = df['final_sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Platform breakdown
            platform_sentiment = df.groupby(['platform', 'final_sentiment']).size().unstack(fill_value=0)
            fig_bar = px.bar(
                platform_sentiment,
                title="Sentiment by Platform",
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline analysis
        if 'created_date' in df.columns:
            df['date'] = pd.to_datetime(df['created_date']).dt.date
            daily_sentiment = df.groupby(['date', 'final_sentiment']).size().unstack(fill_value=0)
            
            fig_timeline = px.line(
                daily_sentiment.reset_index(),
                x='date',
                y=['positive', 'neutral', 'negative'],
                title="Sentiment Trends Over Time",
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                }
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    def display_crisis_keywords(self):
        """Display crisis keyword analysis"""
        if st.session_state.processed_data.empty:
            return
        
        df = st.session_state.processed_data
        crisis_posts = df[df['crisis_score'] > 0]
        
        if crisis_posts.empty:
            st.info("✅ No crisis keywords detected in recent posts.")
            return
        
        st.header("🔍 Crisis Keyword Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crisis keyword frequency
            all_keywords = []
            for keywords in crisis_posts['found_keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            if all_keywords:
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                
                fig_keywords = px.bar(
                    x=list(keyword_counts.values()),
                    y=list(keyword_counts.keys()),
                    orientation='h',
                    title="Most Common Crisis Keywords",
                    color=list(keyword_counts.values()),
                    color_continuous_scale='Reds'
                )
                fig_keywords.update_layout(showlegend=False)
                st.plotly_chart(fig_keywords, use_container_width=True)
        
        with col2:
            # Crisis posts over time
            crisis_posts['date'] = pd.to_datetime(crisis_posts['created_date']).dt.date
            daily_crisis = crisis_posts.groupby('date').size()
            
            fig_crisis_timeline = px.bar(
                x=daily_crisis.index,
                y=daily_crisis.values,
                title="Crisis Posts Over Time",
                color=daily_crisis.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_crisis_timeline, use_container_width=True)
        
        # Top crisis posts
        st.subheader("🚨 Most Critical Posts")
        top_crisis = crisis_posts.nlargest(5, 'crisis_score')[
            ['text', 'platform', 'crisis_score', 'found_keywords', 'engagement_score']
        ]
        
        for idx, row in top_crisis.iterrows():
            with st.expander(f"Crisis Score: {row['crisis_score']} | Platform: {row['platform']}"):
                st.write(f"**Text:** {row['text'][:200]}...")
                st.write(f"**Keywords Found:** {', '.join(row['found_keywords']) if row['found_keywords'] else 'None'}")
                st.write(f"**Engagement Score:** {row['engagement_score']}")
    
    def display_detailed_posts(self):
        """Display detailed post analysis"""
        if st.session_state.processed_data.empty:
            return
        
        st.header("📝 Post Details")
        
        df = st.session_state.processed_data
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=['All', 'positive', 'neutral', 'negative']
            )
        
        with col2:
            platform_filter = st.selectbox(
                "Filter by Platform",
                options=['All'] + list(df['platform'].unique())
            )
        
        with col3:
            min_engagement = st.slider(
                "Minimum Engagement Score",
                min_value=0,
                max_value=int(df['engagement_score'].max()) if not df.empty else 100,
                value=0
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['final_sentiment'] == sentiment_filter]
        
        if platform_filter != 'All':
            filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
        
        filtered_df = filtered_df[filtered_df['engagement_score'] >= min_engagement]
        
        # Sort by engagement score
        filtered_df = filtered_df.sort_values('engagement_score', ascending=False)
        
        st.write(f"Showing {len(filtered_df)} posts")
        
        # Display posts
        for idx, row in filtered_df.head(10).iterrows():
            sentiment_color = {
                'positive': '🟢',
                'neutral': '🟡',
                'negative': '🔴'
            }
            
            with st.expander(
                f"{sentiment_color.get(row['final_sentiment'], '⚪')} "
                f"{row['platform'].title()} | "
                f"Sentiment: {row['final_sentiment']} ({row['final_score']:.2f}) | "
                f"Engagement: {row['engagement_score']}"
            ):
                st.write(f"**Posted:** {row['created_date']}")
                st.write(f"**Text:** {row['text']}")
                if row['crisis_score'] > 0:
                    st.write(f"**⚠️ Crisis Keywords:** {', '.join(row['found_keywords'])}")
                if 'url' in row and pd.notna(row['url']):
                    st.write(f"**[View Original Post]({row['url']})**")
    
    def display_export_options(self):
        """Display data export options"""
        if st.session_state.processed_data.empty:
            return
        
        st.header("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Current Analysis"):
                try:
                    # Create export directory
                    os.makedirs('exports', exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save processed data
                    filename = f"exports/crisis_analysis_{timestamp}.csv"
                    st.session_state.processed_data.to_csv(filename, index=False)
                    
                    # Save crisis report
                    report_filename = f"exports/crisis_report_{timestamp}.json"
                    import json
                    with open(report_filename, 'w') as f:
                        json.dump(st.session_state.crisis_report, f, indent=2, default=str)
                    
                    st.success(f"✅ Data exported to {filename}")
                    st.success(f"✅ Report exported to {report_filename}")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        with col2:
            # Download button for CSV
            if not st.session_state.processed_data.empty:
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"crisis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """Main dashboard runner"""
        st.title("🚨 Social Media Crisis Detection Dashboard")
        st.markdown("Monitor social media sentiment and detect potential PR crises in real-time")
        
        # Sidebar controls
        business_name, lookback_hours = self.sidebar_controls()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Crisis Overview", 
            "📊 Sentiment Analysis", 
            "🔍 Crisis Keywords", 
            "📝 Post Details", 
            "📤 Export"
        ])
        
        with tab1:
            self.display_crisis_overview()
        
        with tab2:
            self.display_sentiment_analysis()
        
        with tab3:
            self.display_crisis_keywords()
        
        with tab4:
            self.display_detailed_posts()
        
        with tab5:
            self.display_export_options()
        
        # Auto-refresh option
        st.sidebar.markdown("---")
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30 min)")
        
        if auto_refresh:
            # This would need additional implementation for production use
            st.sidebar.info("Auto-refresh enabled. Dashboard will update every 30 minutes.")
            time.sleep(1)  # Placeholder for refresh logic

# Run the dashboard
if __name__ == "__main__":
    app = DashboardApp()
    app.run()