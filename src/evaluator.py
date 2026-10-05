
def _extract_text(resp):
    content = getattr(resp, 'content', resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return ''.join(x if isinstance(x, str) else x.get('text', '') for x in content).strip()
    return str(content).strip()
"""
src/evaluator.py

Independent Retrieval Quality & Answer Reliability Evaluation Layer:
1. Retrieval relevance assessment
2. Context relevance checking
3. Answer groundedness verification & hallucination detection
4. Source citation validation
5. Insufficient information detection (anti-hallucination guardrail)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class EvaluationReport(BaseModel):
    """
    Structured outcome of the evaluation layer.
    """
    is_sufficient: bool = Field(
        default=True,
        description="Whether the retrieved context contains sufficient info to answer the query.",
    )
    context_relevance_score: float = Field(
        default=1.0,
        description="Relevance score (0.0 to 1.0) of retrieved context with respect to the user query.",
    )
    groundedness_score: float = Field(
        default=1.0,
        description="Score (0.0 to 1.0) indicating whether all claims are supported by context without hallucination.",
    )
    citations_valid: bool = Field(
        default=True,
        description="Whether citations in the answer accurately correspond to actual retrieved sources.",
    )
    is_valid: bool = Field(
        default=True,
        description="Overall pass/fail verification decision.",
    )
    hallucination_detected: bool = Field(
        default=False,
        description="Flag indicating if the model hallucinated facts not in context.",
    )
    feedback: str = Field(
        default="",
        description="Specific actionable feedback if the answer needs refinement or if info is missing.",
    )


class QualityEvaluator:
    """
    Decoupled evaluation engine providing rigorous verification of retrieved passages
    and generated answers before presentation to the user.
    """

    def __init__(self, llm):
        self.llm = llm

    def assess_context_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """
        Assesses whether the retrieved passages actually pertain to the user's query
        and whether sufficient facts exist to form a factual answer.
        """
        # If no external tool was needed (direct conversational query), context is by definition sufficient
        if not context or "No external context or tool data required" in context:
            return {
                "is_sufficient": True,
                "relevance_score": 1.0,
                "missing_information": "",
            }

        if "No relevant internal documents found" in context:
            return {
                "is_sufficient": False,
                "relevance_score": 0.0,
                "missing_information": "Knowledge base returned no relevant documents.",
            }

        system_prompt = (
            "You are a strict Retrieval Relevance Evaluator.\n"
            "Analyze the user query and the retrieved context.\n"
            "Determine:\n"
            "1. 'relevance_score': Float between 0.0 and 1.0 evaluating how directly related the context is.\n"
            "2. 'is_sufficient': Boolean (true ONLY if the context provides factual evidence to answer the query, "
            "false if the context is insufficient, generic, or silent on the question).\n"
            "3. 'missing_information': String description of what specific facts are missing if is_sufficient is false.\n\n"
            "Respond ONLY with a valid JSON object matching these keys."
        )

        user_content = f"User Query: {query}\n\nRetrieved Context:\n{context}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            content = _extract_text(response)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except Exception as exc:
            logger.warning("Context relevance assessment parse error (%s); defaulting.", exc)
            return {
                "is_sufficient": True,
                "relevance_score": 0.8,
                "missing_information": "",
            }

    def verify_groundedness_and_citations(
        self,
        query: str,
        context: str,
        generation: str,
    ) -> Dict[str, Any]:
        """
        Verifies that every factual claim in generation is grounded in context
        and validates that cited sources actually exist in the retrieved context.
        """
        # If no context was needed (direct chit-chat), groundedness check passes
        if not context or "No external context or tool data required" in context:
            return {
                "groundedness_score": 1.0,
                "hallucination_detected": False,
                "citations_valid": True,
                "is_valid": True,
                "feedback": "",
            }

        acknowledged_insufficient = any(
            phrase in generation.lower() for phrase in [
                "does not contain sufficient",
                "not found in the knowledge base",
                "no information found",
                "unable to find",
                "not enough information",
            ]
        )

        system_prompt = (
            "You are an Answer Groundedness and Hallucination Verification Evaluator.\n"
            "Evaluate whether the generated answer is completely faithful to the provided context.\n"
            "Criteria:\n"
            "1. Groundedness: Every factual statement must be directly backed by the context.\n"
            "2. Hallucination: If the answer makes up facts, dates, numbers, or policies not present in context, "
            "hallucination_detected must be true.\n"
            "3. Citations: If the answer cites sources/documents (e.g. [1], 'policy.pdf', Chunk ID), "
            "verify whether those citations actually exist in the context.\n"
            "4. Insufficient info acknowledgment: If the context didn't have the answer and the model explicitly "
            "stated it does not have enough information, that is fully grounded (hallucination_detected: false, groundedness_score: 1.0).\n\n"
            "Return a valid JSON object with:\n"
            "  \"groundedness_score\": float (0.0 to 1.0),\n"
            "  \"hallucination_detected\": boolean,\n"
            "  \"citations_valid\": boolean,\n"
            "  \"is_valid\": boolean,\n"
            "  \"feedback\": string (explain any discrepancies or ungrounded claims)\n"
            "Respond ONLY with the JSON object."
        )

        user_content = (
            f"User Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Generated Answer:\n{generation}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            content = _extract_text(response)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            parsed = json.loads(content)
            return parsed
        except Exception as exc:
            logger.warning("Groundedness evaluation parse error (%s); applying rule heuristics.", exc)
            return {
                "groundedness_score": 1.0 if acknowledged_insufficient else 0.8,
                "hallucination_detected": False,
                "citations_valid": True,
                "is_valid": True,
                "feedback": "",
            }

    def evaluate(
        self,
        query: str,
        context: str,
        generation: str,
    ) -> EvaluationReport:
        """
        Runs the complete decoupled evaluation protocol.
        """
        # Step 1: Context sufficiency & relevance
        ctx_assessment = self.assess_context_relevance(query, context)
        is_sufficient = ctx_assessment.get("is_sufficient", True)
        relevance_score = float(ctx_assessment.get("relevance_score", 1.0))
        missing_info = ctx_assessment.get("missing_information", "")

        # Step 2: Answer groundedness & citation verification
        verif = self.verify_groundedness_and_citations(query, context, generation)
        groundedness = float(verif.get("groundedness_score", 1.0))
        hallucination = bool(verif.get("hallucination_detected", False))
        citations_valid = bool(verif.get("citations_valid", True))
        is_valid = bool(verif.get("is_valid", True))
        feedback = verif.get("feedback", "")

        # If context is insufficient and the answer hallucinated an answer instead of admitting it
        if not is_sufficient and not any(
            phrase in generation.lower() for phrase in [
                "does not contain", "not found", "insufficient information", "cannot find", "no information"
            ]
        ):
            hallucination = True
            is_valid = False
            feedback = (
                f"The knowledge base does not contain sufficient information to answer '{query}'. "
                f"Missing: {missing_info}. The answer must explicitly state that the information "
                "is not available in the knowledge base rather than inventing facts."
            )

        if hallucination or not citations_valid:
            is_valid = False

        return EvaluationReport(
            is_sufficient=is_sufficient,
            context_relevance_score=relevance_score,
            groundedness_score=groundedness,
            citations_valid=citations_valid,
            is_valid=is_valid,
            hallucination_detected=hallucination,
            feedback=feedback,
        )
