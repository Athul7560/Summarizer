from __future__ import annotations

from dataclasses import dataclass

from backend.database import student_repo
from backend.model_service import llm_service
from backend.rag_service import rag_service


@dataclass
class Suggestion:
    student_id: str
    topic: str
    suggestion_type: str
    recommendation: str


class StudyCoachAgent:
    SUGGESTION_TYPES = ("summary", "quiz", "flashcards")

    def get_weak_topics(self, student_id: str, threshold: float = 60.0) -> list[str]:
        topic_scores = student_repo.get_topic_scores(student_id)
        return [row["topic"] for row in topic_scores if float(row["avg_score"]) < threshold]

    def trigger_summary(self, topic: str, depth: str = "concise") -> str:
        task = f"Provide a {depth} study summary"
        return llm_service.generate(task=task, user_input=topic, context=[])

    def generate_quiz(self, topic: str, difficulty: str = "medium") -> str:
        task = f"Create a {difficulty} quiz with 5 MCQs and answer key"
        return llm_service.generate(task=task, user_input=topic, context=[])

    def create_flashcards(self, topic: str) -> str:
        return llm_service.generate(task="Create 8 Q/A flashcards", user_input=topic, context=[])

    def recommend(self, student_id: str) -> Suggestion:
        weak_topics = self.get_weak_topics(student_id)
        declined = student_repo.get_declined_suggestions(student_id)

        for topic in weak_topics:
            for suggestion_type in self.SUGGESTION_TYPES:
                if (topic, suggestion_type) in declined:
                    continue
                message = (
                    f"You are weaker in '{topic}'. I recommend a {suggestion_type} intervention now. "
                    f"Would you like me to generate it?"
                )
                return Suggestion(student_id=student_id, topic=topic, suggestion_type=suggestion_type, recommendation=message)

        return Suggestion(
            student_id=student_id,
            topic="general",
            suggestion_type="review",
            recommendation="No critical weak topic detected. Keep practicing with mixed-topic quizzes.",
        )

    def run_tool(self, suggestion: Suggestion) -> str:
        if suggestion.suggestion_type == "summary":
            return self.trigger_summary(suggestion.topic, depth="detailed")
        if suggestion.suggestion_type == "quiz":
            return self.generate_quiz(suggestion.topic, difficulty="easy")
        if suggestion.suggestion_type == "flashcards":
            return self.create_flashcards(suggestion.topic)
        related = rag_service.retrieve(query=suggestion.topic, student_id=suggestion.student_id, top_k=3)
        return "General review:\n" + "\n\n".join(related)


study_coach_agent = StudyCoachAgent()
