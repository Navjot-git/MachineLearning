from agents.sentiment_agent import SentimentAgent

class FeedbackAgent:
    def __init__(self, faq_agent, escalation_agent):
        # List to store feedback
        self.feedback_log = []
        self.sentiment_agent = SentimentAgent()
        self.faq_agent = faq_agent # Reference to FAQ agent for checking escalation
        self.escalation_agent = escalation_agent

    def collect_feedback(self, user_feedback: str):
        # Analyze the sentiment of the feedback
        sentiment = self.sentiment_agent.analyze_sentiment(user_feedback)

        # If feedback is negative and FAQ Agent has not escalated, escalate the issue
        if sentiment['compound'] < -0.5 and not self.faq_agent.has_escalated():
            self.escalation_agent.escalate_issue(user_feedback)

        # Add feedback to the log along with sentiment analysis
        self.feedback_log.append({"feedback": user_feedback, "sentiment": sentiment})
        return "Thank you for your feedback!"

    def get_all_feedback(self):
        # Return all collected feedback
        return self.feedback_log