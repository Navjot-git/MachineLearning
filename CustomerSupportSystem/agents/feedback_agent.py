class FeedbackAgent:
    def __init__(self):
        # List to store feedback
        self.feedback_log = []

    def collect_feedback(self, user_feedback: str):
        # Add feedback to the log
        self.feedback_log.append(user_feedback)
        return "Thank you for your feedback!"

    def get_all_feedback(self):
        # Return all collected feedback
        return self.feedback_log