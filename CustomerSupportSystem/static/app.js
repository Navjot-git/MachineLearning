document.addEventListener("DOMContentLoaded", () => {
    // Select the cards and input elements
    const faqCard = document.querySelector(".faq");
    const escalationCard = document.querySelector(".escalation");
    const feedbackCard = document.querySelector(".feedback");
    const queryInput = document.getElementById("user-query");
    const feedbackInput = document.getElementById("feedback-input");

    // Initially, hide the escalation and feedback cards
    escalationCard.style.opacity = "0.3";
    feedbackCard.style.opacity = "0.3";

    // Function to handle query submission
    async function submitQuery(question) {
        try {
            // Fetch the answer and status from the backend
            const response = await fetch(`/faq?question=${encodeURIComponent(question)}`);
            const data = await response.json();

            // Check the status and update the cards accordingly
            if (data.status === "resolved") {
                faqCard.style.borderColor = "green";  // FAQ resolved
                feedbackCard.style.opacity = "1";  // Activate feedback collection
            } else if (data.status === "escalated") {
                faqCard.style.borderColor = "red";  // FAQ escalated
                escalationCard.style.opacity = "1";  // Activate escalation card
                feedbackCard.style.opacity = "1";  // Activate feedback collection after escalation
            }
        } catch (error) {
            console.error("Error fetching FAQ data:", error);
        }
    }

    // Event listener for query submission
    document.getElementById("submit-query").addEventListener("click", () => {
        const userQuestion = queryInput.value;
        if (userQuestion) {
            submitQuery(userQuestion);  // Call the submitQuery function with user input
        } else {
            alert("Please enter a question before submitting.");
        }
    });

    // Add event listener for feedback submission
    document.getElementById("submit-feedback").addEventListener("click", async () => {
        const feedbackText = feedbackInput.value;

        try {
            const response = await fetch("/feedback", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ feedback: feedbackText }),
            });

            const data = await response.json();
            alert(data.message + " Sentiment: " + JSON.stringify(data.sentiment));
        } catch (error) {
            console.error("Error submitting feedback:", error);
        }
    });
});
