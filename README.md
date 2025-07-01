# 🚨 Social Cubes
Advanced Social Media Crisis Detection System

## 📖 What is Social Cubes?

Social Cubes is an intelligent Social Media Crisis Detection System that provides automated monitoring and early warning capabilities for potential PR crises. It enables businesses to respond proactively before issues escalate and damage brand reputation through real-time sentiment analysis and crisis prediction.

![Screenshot 2025-07-01 143206](https://github.com/user-attachments/assets/1de40bba-ba84-40ce-9294-9e33af251736)

🎯 Problem Statement

The Challenge:
1. 73% of business crises originate from social media discussions
2. Average response time to social media crises is 4-6 hours
3. 1 hour delay in crisis response can result in 5x increase in negative sentiment spread
4. Manual monitoring is inefficient and misses early warning signs

Real-World Impact:
1. United Airlines lost $1.4B in market value after a social media crisis
2. Chipotle's stock dropped 42% following food safety issues amplified on social media
3. Small businesses can lose 22% of customers after a single negative viral incident

💡 Solution Overview
Social Cubes provides comprehensive crisis detection through three core components:
🔍 Real-Time Monitoring
>>Automated social media data collection from Reddit and Twitter

>>24/7 monitoring with customizable keywords and sentiment tracking

>>Multi-platform aggregation for comprehensive coverage

🧠 Intelligent Crisis Detection

>>Advanced sentiment analysis using VADER and TextBlob

>>Crisis keyword detection with customizable risk categories

>>Engagement-weighted scoring for viral potential assessment

>>Multi-factor crisis scoring algorithm

![image](https://github.com/user-attachments/assets/6085a4ad-f7db-4cfd-9f2d-2ae246f570c4)


📊 Actionable Insights Dashboard

>>Real-time crisis level indicators (Low/Medium/High)

>>Automated alert generation with specific recommendations

>>Trend analysis and sentiment tracking over time

>>Exportable reports for stakeholder communication

![image](https://github.com/user-attachments/assets/e955b9c0-4d0e-45e3-880a-81c2947da7c2)


## 🏗️ Technical Architecture

📱 Data Sources (Reddit, Twitter)

           ↓
           
🔄 Data Collection Layer (Python APIs)

           ↓
           
⚙️ Processing Engine (Sentiment Analysis, Crisis Detection)

           ↓          
📈 Real-time Dashboard (Streamlit)

           ↓
           
🚨 Alert System & Reporting

## 🛠️ Tech Stack

Data Collection: Reddit API (PRAW), Twitter API v2

AI/ML: VADER Sentiment, TextBlob, scikit-learn

Visualization: Streamlit, Plotly, Pandas

Deployment: Docker containerization, cloud-ready

## 🚀 Getting Started
### Prerequisites:

> Python 3.8+

> Reddit API credentials

> Twitter API v2 credentials

### Installation

#### Clone the repository

```clone https://github.com/yourusername/social-cubes.git```

```cd social_cubes```

#### Install dependencies

```install -r requirements.txt```

#### Configure API credentials

```config/config.example.py config/config.py #Edit config.py with your API credentials```


####Run the application

```streamlit run app.py```

## 📋 Features

✅ Multi-platform Monitoring - Reddit, Twitter integration

✅ Real-time Sentiment Analysis - VADER + TextBlob powered

✅ Crisis Scoring Algorithm - Multi-factor risk assessment

✅ Interactive Dashboard - Live updates and visualizations

✅ Customizable Alerts - Configurable thresholds and keywords

✅ Export Capabilities - Generate reports for stakeholders

✅ Trend Analysis - Historical data and pattern recognition

## 📈 Usage

Setup Keywords: Configure monitoring keywords for your brand

Set Thresholds: Define crisis severity levels

Monitor Dashboard: Track real-time sentiment and engagement

Receive Alerts: Get notified when potential crises are detected

Export Reports: Generate comprehensive crisis reports

## 🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer
This project is designed for educational and professional demonstration purposes. Ensure proper authorization and follow organizational security policies before implementing in production environments. The author is not responsible for any misuse of this system.
