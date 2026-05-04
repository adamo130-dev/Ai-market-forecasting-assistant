#!/usr/bin/env python3
"""
app.py - Entry point for Hugging Face Spaces
Multi-Agent Stock Market Forecasting Application
"""

import sys
import os
import subprocess
import streamlit as st

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def install_requirements():
    """Install required packages if not already installed"""
    required_packages = [
        'streamlit',
        'yfinance',
        'pandas',
        'numpy',
        'plotly',
        'requests',
        'textblob',
        'scikit-learn',
        'feedparser',
        'beautifulsoup4',
        'reportlab',
        'groq'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Main entry point for Hugging Face Spaces"""
    try:
        # Install requirements if needed
        install_requirements()
        
        # Import and run the stock application
        from stock import main as stock_main
        
        # Set up the Streamlit page configuration
        st.set_page_config(
            page_title="🤖 Multi-Agent Stock Market Forecasting",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add Hugging Face specific styling and information
        st.markdown("""
        <style>
        .huggingface-header {
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            font-weight: bold;
        }
        .space-info {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            border-left: 4px solid #4ecdc4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state - FIXED: Added missing ai_chat_history initialization
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
        
        # Initialize ai_chat_history if not exists
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # Initialize other common session state variables that might be needed
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
            
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = None
            
        if 'groq_api_key' not in st.session_state:
            st.session_state.groq_api_key = None
            
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
            
        if 'prediction_data' not in st.session_state:
            st.session_state.prediction_data = None
            
        if 'news_data' not in st.session_state:
            st.session_state.news_data = None
            
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        
        # Add Hugging Face Space header
        st.markdown("""
        <div class="huggingface-header">
            🤗 Hugging Face Space: Multi-Agent Stock Market Forecasting
        </div>
        """, unsafe_allow_html=True)
        
        # Add space information
        with st.expander("ℹ️ About This Hugging Face Space"):
            st.markdown("""
            <div class="space-info">
            <h4>🚀 Welcome to Multi-Agent Stock Market Forecasting!</h4>
            
            This Hugging Face Space provides a comprehensive stock market analysis platform using multiple AI agents:
            
            <ul>
            <li><strong>📰 NewsFetcherAgent:</strong> Gathers recent financial news</li>
            <li><strong>📊 MarketDataAgent:</strong> Retrieves historical stock data</li>
            <li><strong>🧠 InsightGeneratorAgent:</strong> Analyzes news sentiment</li>
            <li><strong>⚙️ PreprocessingAgent:</strong> Calculates technical indicators</li>
            <li><strong>🔮 PredictionAgent:</strong> ML-powered price predictions</li>
            <li><strong>🤖 LLMSummarizerAgent:</strong> AI-generated insights</li>
            <li><strong>📄 ReportAgent:</strong> Multi-format report generation</li>
            </ul>
            
            <h5>🛠️ Features:</h5>
            <ul>
            <li>Real-time stock data analysis</li>
            <li>Technical indicator calculations (RSI, MACD, Bollinger Bands)</li>
            <li>Machine learning price predictions</li>
            <li>News sentiment analysis</li>
            <li>Interactive charts and visualizations</li>
            <li>Downloadable reports (HTML, PDF, JSON)</li>
            <li>AI-powered market insights</li>
            </ul>
            
            <h5>🔧 How to Use:</h5>
            <ol>
            <li>Enter a stock symbol (e.g., AAPL, GOOGL, TSLA) in the sidebar</li>
            <li>Configure analysis parameters (period, prediction days)</li>
            <li>Optionally add your Groq API key for AI-powered insights</li>
            <li>Click "🚀 Run Analysis" to start the multi-agent pipeline</li>
            <li>Explore results in different tabs</li>
            <li>Download comprehensive reports</li>
            </ol>
            
            <p><strong>⚠️ Disclaimer:</strong> This application is for educational purposes only. 
            Not financial advice. Always consult qualified financial advisors before making investment decisions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Run the main stock application
        stock_main()
        
        # Add footer with additional information
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🤗 Hugging Face Space**")
            st.markdown("Powered by Streamlit")
        
        with col2:
            st.markdown("**📊 Data Sources**")
            st.markdown("Yahoo Finance, RSS Feeds")
        
        with col3:
            st.markdown("**🤖 AI Integration**")
            st.markdown("Groq API, scikit-learn")
        
    except ImportError as e:
        st.error(f"Import Error: {str(e)}")
        st.markdown("""
        **Installation Issue Detected**
        
        It seems some required packages are not installed. This Space requires:
        - streamlit
        - yfinance
        - pandas
        - numpy
        - plotly
        - scikit-learn
        - textblob
        - feedparser
        - beautifulsoup4
        - reportlab
        - groq
        
        Please ensure all dependencies are properly installed.
        """)
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.markdown("""
        **Something went wrong!**
        
        Please try refreshing the page or contact the space maintainer if the issue persists.
        """)

if __name__ == "__main__":
    main()
