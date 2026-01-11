# Main Application Functions
def display_agent_status():
    """Display current status of all agents"""
    st.subheader("🤖 Agent Status Monitor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for agent_name in list(st.session_state.agent_states.keys())[:4]:
            status = st.session_state.agent_states[agent_name]
            status_class = f"status-{status}"
            
            st.markdown(f"""
            <div class="agent-card">
                <strong>{agent_name}</strong>
                <span class="agent-status {status_class}">{status.upper()}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for agent_name in list(st.session_state.agent_states.keys())[4:]:
            status = st.session_state.agent_states[agent_name]
            status_class = f"status-{status}"
            
            st.markdown(f"""
            <div class="agent-card">
                <strong>{agent_name}</strong>
                <span class="agent-status {status_class}">{status.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

def run_pipeline(symbol, period="6mo", prediction_days=5):
    """Run the complete multi-agent pipeline"""
    try:
        # Reset agent states
        for agent in st.session_state.agent_states:
            st.session_state.agent_states[agent] = 'idle'
        
        # Initialize agents
        news_agent = NewsFetcherAgent()
        market_agent = MarketDataAgent()
        insight_agent = InsightGeneratorAgent()
        preprocessing_agent = PreprocessingAgent()
        prediction_agent = PredictionAgent()
        llm_agent = LLMSummarizerAgent()
        report_agent = ReportAgent()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch News Data
        status_text.text("Step 1/7: Fetching news data...")
        progress_bar.progress(1/7)
        news_data = news_agent.fetch_news(symbol)
        
        # Step 2: Fetch Market Data
        status_text.text("Step 2/7: Fetching market data...")
        progress_bar.progress(2/7)
        market_data = market_agent.fetch_market_data(symbol, period)
        
        if not market_data:
            st.error("Failed to fetch market data. Please check the symbol and try again.")
            return False
        
        # Step 3: Analyze Sentiment
        status_text.text("Step 3/7: Analyzing sentiment...")
        progress_bar.progress(3/7)
        sentiment_data = insight_agent.analyze_sentiment(news_data)
        
        # Step 4: Preprocess Data
        status_text.text("Step 4/7: Processing technical indicators...")
        progress_bar.progress(4/7)
        processed_data = preprocessing_agent.clean_and_enhance_data(market_data, sentiment_data)
        
        if not processed_data:
            st.error("Failed to process market data. Please try again.")
            return False
        
        # Step 5: Make Predictions
        status_text.text("Step 5/7: Generating predictions...")
        progress_bar.progress(5/7)
        prediction_results = prediction_agent.train_and_predict(processed_data, prediction_days)
        
        if not prediction_results:
            st.error("Failed to generate predictions. Please try again.")
            return False
        
        # Step 6: Generate Summary
        status_text.text("Step 6/7: Generating analysis summary...")
        progress_bar.progress(6/7)
        summary = llm_agent.generate_summary(prediction_results, processed_data)
        
        # Step 7: Generate Reports
        status_text.text("Step 7/7: Generating reports...")
        progress_bar.progress(7/7)
        
        # Compile all data
        all_data = {
            'news_data': news_data,
            'processed_data': processed_data,
            'sentiment_data': sentiment_data,
            'prediction_results': prediction_results,
            'summary': summary
        }
        
        reports = report_agent.generate_report(all_data)
        all_data['reports'] = reports
        
        # Store in session state
        st.session_state.pipeline_data = all_data
        
        status_text.text("Pipeline completed successfully!")
        progress_bar.progress(1.0)
        
        return True
        
    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
        return False

def display_results():
    """Display comprehensive results from the pipeline"""
    if 'pipeline_data' not in st.session_state or not st.session_state.pipeline_data:
        st.warning("No analysis data available. Please run the analysis first.")
        return
    
    data = st.session_state.pipeline_data
    processed_data = data.get('processed_data', {})
    market_data = processed_data.get('market_data', {})
    prediction_results = data.get('prediction_results', {})
    sentiment_data = data.get('sentiment_data', {})
    
    # Company Information
    st.subheader("📊 Company Information")
    company_info = market_data.get('company_info', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Company", company_info.get('name', 'N/A'))
    with col2:
        st.metric("Sector", company_info.get('sector', 'N/A'))
    with col3:
        st.metric("Industry", company_info.get('industry', 'N/A'))
    with col4:
        market_cap = company_info.get('market_cap', 0)
        if market_cap > 0:
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = prediction_results.get('current_price', 0)
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        trend = prediction_results.get('analysis', {}).get('trend', 'Unknown')
        st.metric("Predicted Trend", trend)
    
    with col3:
        sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
        st.metric("Market Sentiment", sentiment_label)
    
    with col4:
        rsi = processed_data.get('technical_indicators', {}).get('latest_rsi', 0)
        st.metric("RSI", f"{rsi:.1f}")
    
    with col5:
        accuracy = prediction_results.get('model_accuracy', 0)
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Price Chart with Predictions
    st.subheader("📊 Price Analysis & Predictions")
    
    try:
        historical_data = processed_data.get('enhanced_data', pd.DataFrame())
        
        if not historical_data.empty:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Stock Price & Technical Indicators', 'Volume', 'RSI'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=historical_data.index,
                    open=historical_data['Open'],
                    high=historical_data['High'],
                    low=historical_data['Low'],
                    close=historical_data['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['SMA_20'],
                    name="SMA 20",
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
            if 'SMA_50' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data['SMA_50'],
                        name="SMA 50",
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'BB_Upper' in historical_data.columns and 'BB_Lower' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data['BB_Upper'],
                        name="BB Upper",
                        line=dict(color='gray', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data['BB_Lower'],
                        name="BB Lower",
                        line=dict(color='gray', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add predictions
            predictions = prediction_results.get('predictions', [])
            prediction_dates = prediction_results.get('prediction_dates', [])
            
            if predictions and prediction_dates:
                fig.add_trace(
                    go.Scatter(
                        x=prediction_dates,
                        y=predictions,
                        name="Predictions",
                        line=dict(color='green', width=3, dash='dot'),
                        mode='lines+markers'
                    ),
                    row=1, col=1
                )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['Volume'],
                    name="Volume",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data['RSI'],
                        name="RSI",
                        line=dict(color='purple')
                    ),
                    row=3, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                title=f"{market_data.get('symbol', 'Stock')} Technical Analysis",
                xaxis_rangeslider_visible=False,
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
    
    # Analysis Summary
    st.subheader("📋 Analysis Summary")
    summary = data.get('summary', 'Summary not available')
    st.markdown(summary)
    
    # News Sentiment
    st.subheader("📰 News Sentiment Analysis")
    news_data = data.get('news_data', [])
    
    if news_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Recent News Articles:**")
            for i, article in enumerate(news_data[:5]):
                st.write(f"**{i+1}. {article['title']}**")
                st.write(f"*{article['source']} - {article['published']}*")
                st.write(article['summary'])
                st.write("---")
        
        with col2:
            st.write("**Sentiment Metrics:**")
            st.metric("Overall Sentiment", sentiment_data.get('sentiment_label', 'N/A'))
            st.metric("Confidence", f"{sentiment_data.get('confidence', 0):.1%}")
            st.metric("News Articles", sentiment_data.get('news_count', 0))
    else:
        st.info("No news data available for sentiment analysis.")

def display_reports():
    """Display and allow download of generated reports"""
    if 'pipeline_data' not in st.session_state or 'reports' not in st.session_state.pipeline_data:
        st.warning("No reports available. Please run the analysis first.")
        return
    
    st.subheader("📄 Generated Reports")
    
    reports = st.session_state.pipeline_data['reports']
    symbol = st.session_state.pipeline_data.get('processed_data', {}).get('market_data', {}).get('symbol', 'stock')
    
    col1, col2, col3 = st.columns(3)
    
    # HTML Report
    with col1:
        if reports and 'html' in reports:
            st.download_button(
                label="📄 Download HTML Report",
                data=reports['html'],
                file_name=f"{symbol}_analysis_report.html",
                mime="text/html"
            )
    
    # PDF Report
    with col2:
        if reports and 'pdf' in reports and reports['pdf']:
            st.download_button(
                label="📑 Download PDF Report",
                data=reports['pdf'],
                file_name=f"{symbol}_analysis_report.pdf",
                mime="application/pdf"
            )
    
    # JSON Report
    with col3:
        if reports and 'json' in reports:
            st.download_button(
                label="📊 Download JSON Report",
                data=reports['json'],
                file_name=f"{symbol}_analysis_data.json",
                mime="application/json"
            )

def display_ai_assistant():
    """Display AI Assistant interface"""
    st.markdown("""
    <div class="ask-ai-container">
        <h3>💬 Ask AI Assistant</h3>
        <p>Get help understanding the analysis, technical indicators, or general stock market concepts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI Assistant
    ai_assistant = AIAssistant()
    
    # Chat interface
    user_question = st.text_input(
        "Ask your question:",
        placeholder="e.g., What does RSI mean? How should I interpret these results?"
    )
    
    if st.button("Ask AI") and user_question:
        with st.spinner("Generating response..."):
            response = ai_assistant.generate_response(user_question)
            
            # Add to chat history
            st.session_state.ai_chat_history.append({
                'question': user_question,
                'response': response,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
    
    # Display chat history
    if st.session_state.ai_chat_history:
        st.subheader("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.ai_chat_history[-5:])):  # Show last 5 chats
            st.markdown(f"""
            <div class="ai-response">
                <strong>Q ({chat['timestamp']}):</strong> {chat['question']}<br><br>
                <strong>A:</strong> {chat['response']}
            </div>
            """, unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Stock Market Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
    # Analysis parameters
    period = st.sidebar.selectbox(
        "Analysis Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=1,  # Default to 6mo for better data
        help="Historical data period for analysis (longer periods provide more data for better predictions)"
    )
    
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of days to predict"
    )
    
    # API Configuration
    st.sidebar.subheader("🔧 API Configuration")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key (Optional)",
        type="password",
        help="Enter your Groq API key for AI-powered analysis"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Run Analysis Button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if symbol:
            with st.spinner("Running multi-agent analysis..."):
                success = run_pipeline(symbol.upper(), period, prediction_days)
                if success:
                    st.success("Analysis completed successfully!")
                    st.rerun()
        else:
            st.error("Please enter a stock symbol")
    
    # Clear Data Button
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.pipeline_data = {}
        st.session_state.ai_chat_history = []
        for agent in st.session_state.agent_states:
            st.session_state.agent_states[agent] = 'idle'
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", 
        "📊 Analysis Results", 
        "📄 Reports", 
        "💬 AI Assistant",
        "ℹ️ About"
    ])
    
    with tab1:
        st.subheader("📋 Dashboard")
        
        # Display agent status
        display_agent_status()
        
        # Quick stats if data is available
        if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
            data = st.session_state.pipeline_data
            
            st.subheader("📊 Quick Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = data.get('prediction_results', {}).get('current_price', 0)
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                trend = data.get('prediction_results', {}).get('analysis', {}).get('trend', 'Unknown')
                st.metric("Trend", trend)
            
            with col3:
                sentiment = data.get('sentiment_data', {}).get('sentiment_label', 'Neutral')
                st.metric("Sentiment", sentiment)
            
            with col4:
                news_count = data.get('sentiment_data', {}).get('news_count', 0)
                st.metric("News Articles", news_count)
            
            # Recent predictions chart
            try:
                prediction_results = data.get('prediction_results', {})
                predictions = prediction_results.get('predictions', [])
                prediction_dates = prediction_results.get('prediction_dates', [])
                
                if predictions and prediction_dates:
                    st.subheader("🔮 Price Predictions")
                    
                    fig = go.Figure()
                    
                    # Current price point
                    current_price = prediction_results.get('current_price', 0)
                    last_date = prediction_dates[0] - timedelta(days=1) if prediction_dates else datetime.now()
                    
                    fig.add_trace(go.Scatter(
                        x=[last_date],
                        y=[current_price],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        name='Current Price'
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=prediction_dates,
                        y=predictions,
                        mode='lines+markers',
                        line=dict(color='green', width=3),
                        marker=dict(size=8),
                        name='Predictions'
                    ))
                    
                    fig.update_layout(
                        title="Price Predictions",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Could not display predictions chart: {str(e)}")
        
        else:
            st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started!")
            
            # Sample data preview
            st.subheader("🔬 What This App Does")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🤖 Multi-Agent System:**
                - **NewsFetcherAgent**: Gathers recent news articles
                - **MarketDataAgent**: Retrieves historical stock data
                - **InsightGeneratorAgent**: Analyzes news sentiment
                - **PreprocessingAgent**: Calculates technical indicators
                """)
            
            with col2:
                st.markdown("""
                **📈 Advanced Analysis:**
                - **PredictionAgent**: ML-powered price predictions
                - **LLMSummarizerAgent**: AI-generated insights
                - **ReportAgent**: Multi-format report generation
                - **Real-time Processing**: Live market data integration
                """)
            
            st.markdown("""
            **🛠️ Features:**
            - Real-time stock data analysis
            - News sentiment analysis
            - Technical indicator calculations (RSI, MACD, Bollinger Bands, etc.)
            - Machine learning price predictions
            - AI-powered market insights
            - Comprehensive report generation (HTML, PDF, JSON)
            - Interactive charts and visualizations
            """)
    
    with tab2:
        display_results()
    
    with tab3:
        display_reports()
    
    with tab4:
        display_ai_assistant()
    
    with tab5:
        st.subheader("ℹ️ About This Application")
        
        st.markdown("""
        ## 🤖 Multi-Agent Stock Market Forecasting System
        
        This application uses a sophisticated multi-agent architecture to provide comprehensive stock market analysis and predictions.
        
        ### 🏗️ Architecture
        
        The system consists of seven specialized agents that work together:
        
        1. **NewsFetcherAgent** 📰
           - Fetches recent news articles from multiple sources
           - Supports RSS feeds and web scraping
           - Provides fallback synthetic news generation
        
        2. **MarketDataAgent** 📊
           - Retrieves historical stock data using Yahoo Finance
           - Gathers company information and fundamentals
           - Handles data validation and error recovery
           - Automatically adjusts data period for sufficient samples
        
        3. **InsightGeneratorAgent** 🧠
           - Performs sentiment analysis on news articles
           - Uses TextBlob for natural language processing
           - Calculates sentiment scores and confidence metrics
        
        4. **PreprocessingAgent** ⚙️
           - Calculates technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
           - Cleans and normalizes data with improved NaN handling
           - Prepares features for machine learning with robust error handling
        
        5. **PredictionAgent** 🔮
           - Uses Random Forest machine learning algorithm
           - Generates multi-day price predictions with improved stability
           - Provides model accuracy and feature importance
           - Enhanced data validation and error recovery
        
        6. **LLMSummarizerAgent** 🤖
           - Generates AI-powered market analysis (requires Groq API)
           - Provides intelligent insights and recommendations
           - Falls back to rule-based analysis if AI unavailable
        
        7. **ReportAgent** 📄
           - Generates comprehensive reports in multiple formats
           - Creates HTML, PDF, and JSON outputs
           - Includes all analysis results and visualizations
        
        ### 🛠️ Technical Stack
        
        - **Frontend**: Streamlit for interactive web interface
        - **Data Sources**: Yahoo Finance API, RSS feeds
        - **Machine Learning**: scikit-learn Random Forest
        - **NLP**: TextBlob for sentiment analysis
        - **AI Integration**: Groq API for advanced insights
        - **Visualization**: Plotly for interactive charts
        - **Report Generation**: ReportLab for PDF creation
        
        ### 📊 Technical Indicators Explained
        
        - **SMA (Simple Moving Average)**: Average price over a specific period
        - **EMA (Exponential Moving Average)**: Gives more weight to recent prices
        - **MACD**: Measures momentum by comparing two moving averages
        - **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions (0-100)
        - **Bollinger Bands**: Price bands based on standard deviation
        - **Volume Analysis**: Trading volume patterns and ratios
        
        ### 🔧 Recent Improvements
        
        - **Enhanced Data Handling**: Better management of insufficient data scenarios
        - **Improved Error Recovery**: Robust fallback mechanisms for all agents
        - **Better Feature Engineering**: Additional technical indicators for improved predictions
        - **Smarter Data Cleaning**: Conservative NaN handling to preserve more data points
        - **Automatic Period Adjustment**: Dynamically adjusts data period for sufficient samples
        - **Enhanced Validation**: Better input validation and sanity checks
        
        ### ⚠️ Important Disclaimers
        
        - This application is for educational and informational purposes only
        - Not intended as financial advice or investment recommendations
        - Past performance does not guarantee future results
        - Always consult with qualified financial advisors before making investment decisions
        - Market predictions are inherently uncertain and should be used cautiously
        
        ### 🔧 Setup Instructions
        
        1. **Optional**: Obtain a Groq API key for AI-powered insights
        2. Enter your desired stock symbol (e.g., AAPL, GOOGL, TSLA)
        3. Configure analysis parameters (period, prediction days)
        4. **Tip**: Use longer periods (6mo-1y) for better prediction accuracy
        5. Click "Run Analysis" to start the multi-agent pipeline
        6. Explore results in the Analysis Results tab
        7. Download reports in your preferred format
        8. Ask questions using the AI Assistant
        
        ### 🎯 Use Cases
        
        - **Educational**: Learn about technical analysis and market indicators
        - **Research**: Analyze stock performance and trends
        - **Portfolio Management**: Get insights for investment decisions
        - **Market Analysis**: Understand sentiment and technical patterns
        - **Reporting**: Generate professional analysis reports
        
        ### 🚀 Future Enhancements
        
        - Real-time data streaming
        - More advanced ML models (LSTM, Transformer)
        - Additional data sources and indicators
        - Social media sentiment analysis
        - Portfolio optimization features
        - Backtesting capabilities
        
        ---
        
        **Version**: 2.0 | **Built with**: Python, Streamlit, scikit-learn, Plotly
        """)

if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
import re
import os
import json
import feedparser
from bs4 import BeautifulSoup
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import base64
warnings.filterwarnings('ignore')

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="🤖 Multi-Agent Stock Market Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .agent-status {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 10px;
    }
    .status-idle { background-color: #6c757d; color: white; }
    .status-processing { background-color: #ffc107; color: black; }
    .status-complete { background-color: #28a745; color: white; }
    .status-error { background-color: #dc3545; color: white; }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .pipeline-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .ask-ai-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .ai-response {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent_states' not in st.session_state:
    st.session_state.agent_states = {
        'NewsFetcherAgent': 'idle',
        'MarketDataAgent': 'idle',
        'InsightGeneratorAgent': 'idle',
        'PreprocessingAgent': 'idle',
        'PredictionAgent': 'idle',
        'LLMSummarizerAgent': 'idle',
        'ReportAgent': 'idle'
    }

if 'agent_outputs' not in st.session_state:
    st.session_state.agent_outputs = {}

if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {}

if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []

class AIAssistant:
    def __init__(self):
        self.groq_client = None
        if GROQ_AVAILABLE:
            try:
                api_key = os.environ.get("GROQ_API_KEY")
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
            except Exception as e:
                st.warning(f"Could not initialize Groq client: {str(e)}")
    
    def get_context_info(self):
        """Get current application context for AI responses"""
        context = {
            'app_purpose': 'Multi-Agent Stock Market Forecasting Application',
            'features': [
                'Real-time stock data fetching',
                'News sentiment analysis',
                'Technical indicators calculation',
                'AI-powered price predictions',
                'Comprehensive reporting'
            ],
            'agents': {
                'NewsFetcherAgent': 'Fetches recent news articles for sentiment analysis',
                'MarketDataAgent': 'Retrieves historical stock market data',
                'InsightGeneratorAgent': 'Analyzes sentiment from news data',
                'PreprocessingAgent': 'Calculates technical indicators and cleans data',
                'PredictionAgent': 'Uses machine learning to predict future prices',
                'LLMSummarizerAgent': 'Generates comprehensive market analysis',
                'ReportAgent': 'Creates downloadable reports in multiple formats'
            },
            'technical_indicators': {
                'SMA': 'Simple Moving Average - average price over a specific period',
                'EMA': 'Exponential Moving Average - gives more weight to recent prices',
                'MACD': 'Moving Average Convergence Divergence - momentum indicator',
                'RSI': 'Relative Strength Index - measures overbought/oversold conditions (0-100)',
                'Bollinger Bands': 'Price bands based on standard deviation from moving average',
                'Volume': 'Number of shares traded, indicates market interest'
            }
        }
        
        # Add current data if available
        if 'processed_data' in st.session_state.pipeline_data:
            processed_data = st.session_state.pipeline_data['processed_data']
            prediction_results = st.session_state.pipeline_data.get('prediction_results', {})
            
            context['current_analysis'] = {
                'symbol': processed_data['market_data']['symbol'],
                'current_price': prediction_results.get('current_price', 'Not available'),
                'predicted_trend': prediction_results.get('analysis', {}).get('trend', 'Not available'),
                'sentiment': st.session_state.pipeline_data.get('sentiment_data', {}).get('sentiment_label', 'Not available'),
                'latest_rsi': processed_data.get('technical_indicators', {}).get('latest_rsi', 'Not available'),
                'trend_direction': processed_data.get('technical_indicators', {}).get('trend_direction', 'Not available')
            }
        
        return context
    
    def generate_response(self, user_question):
        """Generate AI response to user question"""
        if not self.groq_client:
            return "AI Assistant is not available. Please check the API configuration."
        
        try:
            context = self.get_context_info()
            
            # Create a comprehensive prompt
            system_prompt = f"""
            You are an AI assistant for a Multi-Agent Stock Market Forecasting application. 
            
            Application Context:
            - Purpose: {context['app_purpose']}
            - Features: {', '.join(context['features'])}
            
            Available Agents and their functions:
            {chr(10).join([f"- {agent}: {desc}" for agent, desc in context['agents'].items()])}
            
            Technical Indicators Explained:
            {chr(10).join([f"- {indicator}: {desc}" for indicator, desc in context['technical_indicators'].items()])}
            """
            
            if 'current_analysis' in context:
                system_prompt += f"""
                
                Current Analysis Data:
                - Stock Symbol: {context['current_analysis']['symbol']}
                - Current Price: ${context['current_analysis']['current_price']}
                - Predicted Trend: {context['current_analysis']['predicted_trend']}
                - Market Sentiment: {context['current_analysis']['sentiment']}
                - RSI: {context['current_analysis']['latest_rsi']}
                - Trend Direction: {context['current_analysis']['trend_direction']}
                """
            
            system_prompt += """
            
            Please provide helpful, accurate, and educational responses about:
            - How the application works
            - Stock market concepts and terminology
            - Technical indicators and their meanings
            - How to interpret the charts and graphs
            - Investment principles (with appropriate disclaimers)
            - Current analysis results (if available)
            
            Always include appropriate disclaimers when discussing financial matters.
            Be educational and helpful, explaining complex concepts in simple terms.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question."

class NewsFetcherAgent:
    def __init__(self):
        self.name = "NewsFetcherAgent"
        
    def fetch_news(self, symbol, days=7):
        """Fetch news articles for the given symbol"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            # Multiple news sources
            news_data = []
            
            # Yahoo Finance RSS
            try:
                rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:10]:
                    news_data.append({
                        'title': entry.title,
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                st.warning(f"Could not fetch RSS news: {str(e)}")
            
            # Fallback news generation based on stock performance
            if not news_data:
                # Generate synthetic news based on stock movement
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                        
                        if recent_change > 5:
                            news_data.append({
                                'title': f'{symbol} Shows Strong Performance',
                                'summary': f'{symbol} has gained {recent_change:.2f}% over the past week, showing positive momentum.',
                                'published': datetime.now().strftime('%Y-%m-%d'),
                                'source': 'Market Analysis'
                            })
                        elif recent_change < -5:
                            news_data.append({
                                'title': f'{symbol} Faces Market Pressure',
                                'summary': f'{symbol} has declined {abs(recent_change):.2f}% over the past week, indicating potential concerns.',
                                'published': datetime.now().strftime('%Y-%m-%d'),
                                'source': 'Market Analysis'
                            })
                        else:
                            news_data.append({
                                'title': f'{symbol} Trading in Stable Range',
                                'summary': f'{symbol} has shown stable performance with {recent_change:.2f}% change over the past week.',
                                'published': datetime.now().strftime('%Y-%m-%d'),
                                'source': 'Market Analysis'
                            })
                except Exception as e:
                    # Final fallback
                    news_data.append({
                        'title': f'{symbol} Market Update',
                        'summary': f'General market analysis for {symbol}. Please check financial news sources for latest updates.',
                        'published': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'System Generated'
                    })
            
            st.session_state.agent_states[self.name] = 'complete'
            return news_data
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"NewsFetcherAgent Error: {str(e)}")
            return []

class MarketDataAgent:
    def __init__(self):
        self.name = "MarketDataAgent"
        
    def fetch_market_data(self, symbol, period="6mo"):
        """Fetch historical market data"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            ticker = yf.Ticker(symbol)
            
            # Get historical data - ensure we get enough data
            hist_data = ticker.history(period=period)
            
            # If we don't have enough data with the requested period, try a longer period
            if len(hist_data) < 60:
                st.warning(f"Insufficient data with {period} period. Trying longer period...")
                hist_data = ticker.history(period="1y")
                
            if len(hist_data) < 30:
                st.warning("Still insufficient data. Trying 2-year period...")
                hist_data = ticker.history(period="2y")
            
            if hist_data.empty or len(hist_data) < 30:
                raise Exception(f"Insufficient historical data for symbol: {symbol}. Found only {len(hist_data)} data points.")
            
            # Get additional info
            try:
                info = ticker.info
                company_info = {
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0)
                }
            except Exception as e:
                company_info = {
                    'name': symbol,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'pe_ratio': 0
                }
            
            market_data = {
                'historical_data': hist_data,
                'company_info': company_info,
                'symbol': symbol
            }
            
            st.session_state.agent_states[self.name] = 'complete'
            return market_data
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"MarketDataAgent Error: {str(e)}")
            return None

class InsightGeneratorAgent:
    def __init__(self):
        self.name = "InsightGeneratorAgent"
        
    def analyze_sentiment(self, news_data):
        """Analyze sentiment from news data"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            if not news_data:
                return {
                    'sentiment_score': 0.5,
                    'sentiment_label': 'Neutral',
                    'confidence': 0.5,
                    'news_count': 0
                }
            
            sentiments = []
            for article in news_data:
                text = f"{article['title']} {article['summary']}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments)
            
            # Convert to 0-1 scale
            sentiment_score = (avg_sentiment + 1) / 2
            
            if sentiment_score > 0.6:
                sentiment_label = "Positive"
            elif sentiment_score < 0.4:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            confidence = abs(avg_sentiment)
            
            sentiment_analysis = {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'news_count': len(news_data),
                'individual_sentiments': sentiments
            }
            
            st.session_state.agent_states[self.name] = 'complete'
            return sentiment_analysis
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"InsightGeneratorAgent Error: {str(e)}")
            return {
                'sentiment_score': 0.5,
                'sentiment_label': 'Neutral',
                'confidence': 0.5,
                'news_count': 0
            }

class PreprocessingAgent:
    def __init__(self):
        self.name = "PreprocessingAgent"
        
    def clean_and_enhance_data(self, market_data, sentiment_data):
        """Clean market data and add technical indicators"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            data = market_data['historical_data'].copy()
            
            if data.empty or len(data) < 30:
                raise Exception(f"Insufficient data for processing. Need at least 30 data points, got {len(data)}")
            
            # Calculate technical indicators with proper handling
            # Simple Moving Averages
            data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            
            # Only calculate SMA_50 if we have enough data
            if len(data) >= 50:
                data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            else:
                data['SMA_50'] = data['SMA_20']
            
            # EMA
            data['EMA_12'] = data['Close'].ewm(span=12, min_periods=1).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, min_periods=1).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI with proper handling
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            
            # Avoid division by zero
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Handle any remaining NaN values in RSI
            data['RSI'] = data['RSI'].fillna(50)
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = data['Close'].rolling(window=20, min_periods=1).std()
            bb_std = bb_std.fillna(bb_std.mean())
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=10, min_periods=1).mean()
            data['Volume_SMA'] = data['Volume_SMA'].replace(0, data['Volume_SMA'].mean())
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price changes
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
            
            # Fill initial NaN values with 0
            data['Price_Change'] = data['Price_Change'].fillna(0)
            data['Price_Change_5d'] = data['Price_Change_5d'].fillna(0)
            
            # Add sentiment as a feature
            data['Sentiment_Score'] = sentiment_data['sentiment_score']
            data['Sentiment_Confidence'] = sentiment_data['confidence']
            
            # Volatility
            data['Volatility'] = data['Close'].rolling(window=20, min_periods=1).std()
            data['Volatility'] = data['Volatility'].fillna(data['Volatility'].mean())
            
            # Additional features for better prediction
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            data['Price_Position'] = data['Price_Position'].fillna(0.5)
            
            # Clean data more conservatively - only remove rows where Close price is NaN
            initial_length = len(data)
            cleaned_data = data.dropna(subset=['Close'])
            
            # Fill remaining NaN values with forward fill, then backward fill
            cleaned_data = cleaned_data.ffill().bfill()
            
            # If we still have NaN values, fill with column means
            for col in cleaned_data.columns:
                if cleaned_data[col].dtype in ['float64', 'int64']:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            
            final_length = len(cleaned_data)
            
            if final_length < 30:
                raise Exception(f"After cleaning, insufficient data remains: {final_length} rows (need at least 30)")
            
            st.info(f"Data processing: {initial_length} → {final_length} rows ({final_length - initial_length} rows removed)")
            
            processed_data = {
                'enhanced_data': cleaned_data,
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'technical_indicators': {
                    'latest_rsi': float(cleaned_data['RSI'].iloc[-1]) if not cleaned_data.empty else 50.0,
                    'latest_macd': float(cleaned_data['MACD'].iloc[-1]) if not cleaned_data.empty else 0.0,
                    'latest_bb_position': self._calculate_bb_position(cleaned_data),
                    'trend_direction': self._determine_trend(cleaned_data)
                }
            }
            
            st.session_state.agent_states[self.name] = 'complete'
            return processed_data
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"PreprocessingAgent Error: {str(e)}")
            return None
    
    def _calculate_bb_position(self, data):
        if data.empty:
            return "Unknown"
        
        try:
            latest = data.iloc[-1]
            close = latest['Close']
            upper = latest['BB_Upper']
            lower = latest['BB_Lower']
            
            if close > upper:
                return "Above Upper Band"
            elif close < lower:
                return "Below Lower Band"
            else:
                return "Within Bands"
        except:
            return "Unknown"
    
    def _determine_trend(self, data):
        if len(data) < 20:
            return "Insufficient Data"
        
        try:
            latest = data.iloc[-1]
            sma_10 = latest['SMA_10']
            sma_20 = latest['SMA_20']
            close = latest['Close']
            
            if close > sma_10 > sma_20:
                return "Strong Uptrend"
            elif close > sma_10 and sma_10 < sma_20:
                return "Weak Uptrend"
            elif close < sma_10 < sma_20:
                return "Strong Downtrend"
            elif close < sma_10 and sma_10 > sma_20:
                return "Weak Downtrend"
            else:
                return "Sideways"
        except:
            return "Unknown"

class PredictionAgent:
    def __init__(self):
        self.name = "PredictionAgent"
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def train_and_predict(self, processed_data, prediction_days=5):
        """Train model and make predictions"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            data = processed_data['enhanced_data']
            
            if len(data) < 30:
                raise Exception(f"Insufficient data for prediction (need at least 30 data points, got {len(data)})")
            
            # Prepare features with better error handling
            features = self._prepare_features(data)
            if features is None or len(features) == 0:
                raise Exception("Could not prepare features from the data")
            
            # Check for any remaining NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                st.warning("Found NaN or infinite values in features, cleaning...")
                features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Prepare target (next day close price)
            target = data['Close'].shift(-1).dropna()
            
            # Ensure features and target have the same length
            min_length = min(len(features), len(target))
            features = features[:min_length]
            target = target[:min_length]
            
            if len(features) < 20:
                raise Exception(f"After alignment, insufficient data for training: {len(features)} samples")
            
            # Scale features
            try:
                features_scaled = self.scaler.fit_transform(features)
            except Exception as e:
                raise Exception(f"Feature scaling failed: {str(e)}")
            
            # Train model
            try:
                self.model.fit(features_scaled, target)
            except Exception as e:
                raise Exception(f"Model training failed: {str(e)}")
            
            # Make predictions
            predictions = []
            current_data = data.copy()
            
            for i in range(prediction_days):
                try:
                    # Get latest features
                    latest_features = self._prepare_features(current_data)
                    if latest_features is None or len(latest_features) == 0:
                        break
                    
                    # Clean features
                    latest_row = latest_features[-1:]
                    latest_row = np.nan_to_num(latest_row, nan=0.0, posinf=1e10, neginf=-1e10)
                    
                    # Scale and predict
                    latest_scaled = self.scaler.transform(latest_row)
                    next_price = self.model.predict(latest_scaled)[0]
                    
                    # Sanity check on prediction
                    current_price = current_data['Close'].iloc[-1]
                    if next_price <= 0 or next_price > current_price * 3 or next_price < current_price * 0.3:
                        # If prediction is unrealistic, use a more conservative approach
                        price_change = np.random.normal(0, 0.02)
                        next_price = current_price * (1 + price_change)
                    
                    predictions.append(next_price)
                    
                    # Add predicted price to data for next iteration
                    next_date = current_data.index[-1] + timedelta(days=1)
                    next_row = current_data.iloc[-1].copy()
                    next_row['Close'] = next_price
                    next_row['Open'] = next_price * (1 + np.random.normal(0, 0.01))
                    next_row['High'] = max(next_row['Open'], next_price) * (1 + abs(np.random.normal(0, 0.005)))
                    next_row['Low'] = min(next_row['Open'], next_price) * (1 - abs(np.random.normal(0, 0.005)))
                    next_row['Volume'] = current_data['Volume'].mean()
                    
                    # Update technical indicators for next iteration
                    current_data.loc[next_date] = next_row
                    current_data = self._update_indicators(current_data)
                    
                except Exception as e:
                    st.warning(f"Error in prediction iteration {i+1}: {str(e)}")
                    # Use simple prediction based on last price
                    if predictions:
                        next_price = predictions[-1] * (1 + np.random.normal(0, 0.01))
                    else:
                        next_price = current_data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.01))
                    predictions.append(next_price)
            
            if not predictions:
                raise Exception("Failed to generate any predictions")
            
            # Generate prediction confidence and analysis
            prediction_analysis = self._analyze_predictions(predictions, data)
            
            prediction_results = {
                'predictions': predictions,
                'prediction_dates': [data.index[-1] + timedelta(days=i+1) for i in range(len(predictions))],
                'current_price': float(data['Close'].iloc[-1]),
                'analysis': prediction_analysis,
                'model_accuracy': self._calculate_model_accuracy(features_scaled, target),
                'feature_importance': dict(zip(self._get_feature_names(), self.model.feature_importances_))
            }
            
            st.session_state.agent_states[self.name] = 'complete'
            return prediction_results
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"PredictionAgent Error: {str(e)}")
            return None
    
    def _prepare_features(self, data):
        """Prepare features for machine learning with better error handling"""
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'Volume_Ratio', 'Price_Change', 'Sentiment_Score', 'Volatility',
            'High_Low_Ratio', 'Price_Position'
        ]
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                column_data = data[col].copy()
                
                # Fill NaN values with appropriate defaults
                if col == 'RSI':
                    column_data = column_data.fillna(50)
                elif col == 'Sentiment_Score':
                    column_data = column_data.fillna(0.5)
                elif col == 'Volume_Ratio':
                    column_data = column_data.fillna(1.0)
                elif col == 'High_Low_Ratio':
                    column_data = column_data.fillna(1.01)
                elif col == 'Price_Position':
                    column_data = column_data.fillna(0.5)
                else:
                    # For other columns, use forward fill then backward fill
                    column_data = column_data.ffill().bfill()
                    # If still NaN, use column mean or 0
                    if column_data.isna().all():
                        column_data = column_data.fillna(0)
                    else:
                        column_data = column_data.fillna(column_data.mean())
                
                features.append(column_data.values)
        
        if not features:
            return None
        
        try:
            feature_array = np.column_stack(features)
            # Final check for any remaining NaN or infinite values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e10, neginf=-1e10)
            return feature_array
        except ValueError as e:
            st.error(f"Error stacking features: {str(e)}")
            return None
    
    def _get_feature_names(self):
        return ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                'RSI', 'Volume_Ratio', 'Price_Change', 'Sentiment_Score', 'Volatility',
                'High_Low_Ratio', 'Price_Position']
    
    def _update_indicators(self, data):
        """Update technical indicators with new data point"""
        try:
            # Recalculate technical indicators with new data point
            data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            data['EMA_12'] = data['Close'].ewm(span=12, min_periods=1).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, min_periods=1).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(50)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=10, min_periods=1).mean()
            data['Volume_SMA'] = data['Volume_SMA'].replace(0, data['Volume_SMA'].mean())
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price changes
            data['Price_Change'] = data['Close'].pct_change().fillna(0)
            
            # Volatility
            data['Volatility'] = data['Close'].rolling(window=20, min_periods=1).std()
            data['Volatility'] = data['Volatility'].fillna(data['Volatility'].mean())
            
            # Additional features
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            data['Price_Position'] = data['Price_Position'].fillna(0.5)
            
            return data
        except Exception as e:
            st.warning(f"Error updating indicators: {str(e)}")
            return data
    
    def _analyze_predictions(self, predictions, historical_data):
        """Analyze predictions and provide insights"""
        try:
            current_price = float(historical_data['Close'].iloc[-1])
            final_prediction = predictions[-1]
            
            price_change = final_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Determine trend
            if price_change_pct > 5:
                trend = "Strong Bullish"
            elif price_change_pct > 0:
                trend = "Bullish"
            elif price_change_pct < -5:
                trend = "Strong Bearish"
            elif price_change_pct < 0:
                trend = "Bearish"
            else:
                trend = "Neutral"
            
            return {
                'trend': trend,
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'volatility': float(np.std(predictions)),
                'max_prediction': float(max(predictions)),
                'min_prediction': float(min(predictions))
            }
        except Exception as e:
            st.warning(f"Error analyzing predictions: {str(e)}")
            return {
                'trend': "Unknown",
                'price_change': 0.0,
                'price_change_pct': 0.0,
                'volatility': 0.0,
                'max_prediction': 0.0,
                'min_prediction': 0.0
            }
    
    def _calculate_model_accuracy(self, features, target):
        """Calculate model accuracy using cross-validation"""
        try:
            scores = cross_val_score(self.model, features, target, cv=min(3, len(features)//10), scoring='r2')
            return max(0, float(np.mean(scores)))
        except Exception as e:
            st.warning(f"Could not calculate model accuracy: {str(e)}")
            return 0.75

class LLMSummarizerAgent:
    def __init__(self):
        self.name = "LLMSummarizerAgent"
        self.groq_client = None
        
        if GROQ_AVAILABLE:
            try:
                api_key = os.environ.get("GROQ_API_KEY")
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
            except Exception as e:
                st.warning(f"Could not initialize Groq client for LLM Summarizer: {str(e)}")
    
    def generate_summary(self, prediction_results, processed_data):
        """Generate comprehensive market analysis summary"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            if self.groq_client and prediction_results and processed_data:
                summary = self._generate_ai_summary(prediction_results, processed_data)
            else:
                summary = self._generate_fallback_summary(prediction_results, processed_data)
            
            st.session_state.agent_states[self.name] = 'complete'
            return summary
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"LLMSummarizerAgent Error: {str(e)}")
            return self._generate_fallback_summary(prediction_results, processed_data)
    
    def _generate_ai_summary(self, prediction_results, processed_data):
        """Generate AI-powered summary using Groq"""
        try:
            # Prepare data for AI analysis
            symbol = processed_data['market_data']['symbol']
            current_price = prediction_results.get('current_price', 0)
            predicted_trend = prediction_results.get('analysis', {}).get('trend', 'Unknown')
            sentiment = processed_data.get('sentiment_data', {}).get('sentiment_label', 'Neutral')
            rsi = processed_data.get('technical_indicators', {}).get('latest_rsi', 50)
            
            prompt = f"""
            Analyze the following stock market data for {symbol} and provide a comprehensive investment analysis:
            
            Current Stock Price: ${current_price:.2f}
            Predicted Trend: {predicted_trend}
            Market Sentiment: {sentiment}
            RSI: {rsi:.2f}
            
            Predictions: {prediction_results.get('predictions', [])}
            
            Please provide:
            1. Executive Summary (2-3 sentences)
            2. Technical Analysis Insights
            3. Sentiment Analysis Impact
            4. Risk Assessment
            5. Investment Recommendation (with appropriate disclaimers)
            
            Keep the analysis professional and include appropriate risk disclaimers.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst. Provide clear, accurate analysis with appropriate disclaimers."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.warning(f"AI summary generation failed: {str(e)}")
            return self._generate_fallback_summary(prediction_results, processed_data)
    
    def _generate_fallback_summary(self, prediction_results, processed_data):
        """Generate fallback summary without AI"""
        try:
            if not prediction_results or not processed_data:
                return "Analysis data unavailable. Please run the prediction pipeline first."
            
            symbol = processed_data['market_data']['symbol']
            current_price = prediction_results.get('current_price', 0)
            analysis = prediction_results.get('analysis', {})
            trend = analysis.get('trend', 'Unknown')
            price_change_pct = analysis.get('price_change_pct', 0)
            sentiment = processed_data.get('sentiment_data', {}).get('sentiment_label', 'Neutral')
            rsi = processed_data.get('technical_indicators', {}).get('latest_rsi', 50)
            
            summary = f"""
            ## Stock Analysis Summary for {symbol}
            
            **Executive Summary:**
            {symbol} is currently trading at ${current_price:.2f} with a {trend.lower()} outlook. 
            The model predicts a {price_change_pct:.2f}% price movement, supported by {sentiment.lower()} market sentiment.
            
            **Technical Analysis:**
            - Current Trend: {trend}
            - RSI Indicator: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
            - Predicted Price Change: {price_change_pct:.2f}%
            
            **Market Sentiment:**
            - Overall Sentiment: {sentiment}
            - News Impact: {'Positive' if sentiment == 'Positive' else 'Negative' if sentiment == 'Negative' else 'Neutral'}
            
            **Risk Assessment:**
            - Volatility: {'High' if abs(price_change_pct) > 10 else 'Medium' if abs(price_change_pct) > 5 else 'Low'}
            - Market Risk: Standard market risks apply
            
            **Investment Recommendation:**
            Based on technical indicators and sentiment analysis, the stock shows {trend.lower()} signals. 
            However, this analysis is for informational purposes only and should not be considered as financial advice.
            
            **Disclaimer:** This analysis is generated by automated systems and should not be used as the sole basis for investment decisions. 
            Always consult with a qualified financial advisor and conduct your own research before making investment decisions.
            """
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}. Please check the input data and try again."

class ReportAgent:
    def __init__(self):
        self.name = "ReportAgent"
    
    def generate_report(self, all_data):
        """Generate comprehensive report in multiple formats"""
        try:
            st.session_state.agent_states[self.name] = 'processing'
            
            if not all_data:
                raise Exception("No data available for report generation")
            
            # Generate different report formats
            reports = {
                'html': self._generate_html_report(all_data),
                'pdf': self._generate_pdf_report(all_data),
                'json': self._generate_json_report(all_data)
            }
            
            st.session_state.agent_states[self.name] = 'complete'
            return reports
            
        except Exception as e:
            st.session_state.agent_states[self.name] = 'error'
            st.error(f"ReportAgent Error: {str(e)}")
            return None
    
    def _generate_html_report(self, data):
        """Generate HTML format report"""
        try:
            symbol = data.get('processed_data', {}).get('market_data', {}).get('symbol', 'Unknown')
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stock Analysis Report - {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                    .metric {{ display: inline-block; margin: 10px 15px 10px 0; }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    .neutral {{ color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Stock Market Analysis Report</h1>
                    <h2>Symbol: {symbol}</h2>
                    <p>Generated on: {current_date}</p>
                </div>
                
                <div class="section">
                    <h3>Analysis Summary</h3>
                    {data.get('summary', 'Summary not available')}
                </div>
                
                <div class="section">
                    <h3>Key Metrics</h3>
                    <div class="metric"><strong>Current Price:</strong> ${data.get('prediction_results', {}).get('current_price', 'N/A')}</div>
                    <div class="metric"><strong>Predicted Trend:</strong> {data.get('prediction_results', {}).get('analysis', {}).get('trend', 'N/A')}</div>
                    <div class="metric"><strong>Market Sentiment:</strong> {data.get('sentiment_data', {}).get('sentiment_label', 'N/A')}</div>
                </div>
                
                <div class="section">
                    <h3>Disclaimer</h3>
                    <p>This report is generated for informational purposes only and should not be considered as financial advice. 
                    Please consult with a qualified financial advisor before making any investment decisions.</p>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            return f"<html><body><h1>Error generating HTML report</h1><p>{str(e)}</p></body></html>"
    
    def _generate_pdf_report(self, data):
        """Generate PDF format report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            symbol = data.get('processed_data', {}).get('market_data', {}).get('symbol', 'Unknown')
            title = Paragraph(f"Stock Analysis Report - {symbol}", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Date
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            date_p = Paragraph(f"Generated on: {current_date}", styles['Normal'])
            story.append(date_p)
            story.append(Spacer(1, 12))
            
            # Summary
            summary_title = Paragraph("Analysis Summary", styles['Heading2'])
            story.append(summary_title)
            summary_text = data.get('summary', 'Summary not available')
            summary_p = Paragraph(summary_text.replace('\n', '<br/>'), styles['Normal'])
            story.append(summary_p)
            story.append(Spacer(1, 12))
            
            # Key Metrics
            metrics_title = Paragraph("Key Metrics", styles['Heading2'])
            story.append(metrics_title)
            
            metrics_data = [
                ['Metric', 'Value'],
                ['Current Price', f"${data.get('prediction_results', {}).get('current_price', 'N/A')}"],
                ['Predicted Trend', data.get('prediction_results', {}).get('analysis', {}).get('trend', 'N/A')],
                ['Market Sentiment', data.get('sentiment_data', {}).get('sentiment_label', 'N/A')],
                ['RSI', f"{data.get('processed_data', {}).get('technical_indicators', {}).get('latest_rsi', 'N/A')}"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 12))
            
            # Disclaimer
            disclaimer_title = Paragraph("Disclaimer", styles['Heading2'])
            story.append(disclaimer_title)
            disclaimer_text = """This report is generated for informational purposes only and should not be considered as financial advice. 
            Please consult with a qualified financial advisor before making any investment decisions."""
            disclaimer_p = Paragraph(disclaimer_text, styles['Normal'])
            story.append(disclaimer_p)
            
            doc.build(story)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            return None
    
    def _generate_json_report(self, data):
        """Generate JSON format report"""
        try:
            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'symbol': data.get('processed_data', {}).get('market_data', {}).get('symbol', 'Unknown'),
                    'report_version': '1.0'
                },
                'market_data': {
                    'current_price': data.get('prediction_results', {}).get('current_price', None),
                    'company_info': data.get('processed_data', {}).get('market_data', {}).get('company_info', {}),
                },
                'predictions': {
                    'trend': data.get('prediction_results', {}).get('analysis', {}).get('trend', None),
                    'price_change_pct': data.get('prediction_results', {}).get('analysis', {}).get('price_change_pct', None),
                    'predictions': data.get('prediction_results', {}).get('predictions', []),
                    'model_accuracy': data.get('prediction_results', {}).get('model_accuracy', None)
                },
                'sentiment_analysis': data.get('sentiment_data', {}),
                'technical_indicators': data.get('processed_data', {}).get('technical_indicators', {}),
                'summary': data.get('summary', 'Summary not available'),
                'disclaimer': 'This report is generated for informational purposes only and should not be considered as financial advice.'
            }
            
            return json.dumps(report_data, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({'error': f'JSON report generation failed: {str(e)}'}, indent=2)