class EscalationAgent:
    def __init__(self):
        self.manual_escalation_log = []
        
    def escalate_issue(self, question):
        # Log the issue for human intervention
        self.manual_escalation_log.append(question)
        
        # Return a message to the user
        return "Your issue has been escalated to a human agent. We'll get back to you shortly."
    
    def get_escalation_log(self):
        # Return the current log of escalated issues (for admin review)
        return self.manual_escalation_log