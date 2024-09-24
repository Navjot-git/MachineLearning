from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAgent:
    def __init__(self):
        # Initialize VADER Sentiment Analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str):
        # Perform sentiment analysis on the provided text
        return self.analyzer.polarity_scores(text)