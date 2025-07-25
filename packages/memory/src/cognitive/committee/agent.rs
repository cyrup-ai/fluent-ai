// src/cognitive/committee/agent.rs
//! Defines the provider-based evaluation agent for the committee.

use serde::Deserialize;

use crate::cognitive::common::models::{Model, ModelType};
use crate::cognitive::common::types::{AgentEvaluation, EvaluationPhase, EvaluationRubric};
use crate::cognitive::mcts::CodeState;
use crate::cognitive::types::CognitiveError;

/// Provider-based evaluation agent
#[derive(Debug, Clone)]
pub struct ProviderEvaluationAgent {
    id: String,
    model: Model,
    perspective: String, // e.g., "performance", "memory", "quality"
}

impl ProviderEvaluationAgent {
    pub async fn new(model_type: ModelType, perspective: &str) -> Result<Self, CognitiveError> {
        let model = Model::create(model_type.clone())
            .await
            .map_err(|e| CognitiveError::ConfigError(e.to_string()))?;

        Ok(Self {
            id: format!("{}_{}_agent", model_type.display_name(), perspective),
            model,
            perspective: perspective.to_string()})
    }

    pub async fn evaluate_with_context(
        &self,
        current_state: &CodeState,
        proposed_action: &str,
        rubric: &EvaluationRubric,
        phase: EvaluationPhase,
        previous_evaluations: Option<&[AgentEvaluation]>,
        steering_feedback: Option<&str>,
    ) -> Result<AgentEvaluation, CognitiveError> {
        let prompt = match phase {
            EvaluationPhase::Initial => {
                self.build_evaluation_prompt(current_state, proposed_action, rubric)
            }
            EvaluationPhase::Review => self.build_review_prompt(
                current_state,
                proposed_action,
                rubric,
                previous_evaluations.unwrap_or(&[]),
            ),
            EvaluationPhase::Refine => self.build_refine_prompt(
                current_state,
                proposed_action,
                rubric,
                previous_evaluations.unwrap_or(&[]),
                steering_feedback.unwrap_or(""),
            ),
            EvaluationPhase::Finalize => self.build_final_prompt(
                current_state,
                proposed_action,
                rubric,
                previous_evaluations.unwrap_or(&[]),
            )};

        let response = self
            .model
            .prompt(&prompt)
            .await
            .map_err(|e| CognitiveError::ApiError(e.to_string()))?;

        self.parse_evaluation_response(&response, proposed_action)
    }

    fn build_review_prompt(
        &self,
        _state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
        others: &[AgentEvaluation],
    ) -> String {
        let other_evals = others.iter()
            .filter(|e| e.agent_id != self.id)
            .map(|e| format!("{}: progress={}, alignment={:.2}, quality={:.2}, risk={:.2}\nReasoning: {}\nSuggestions: {}",
                e.agent_id, e.makes_progress, e.objective_alignment, e.implementation_quality,
                e.risk_assessment, e.reasoning, e.suggested_improvements.join(", ")))
            .collect::<Vec<_>>()
            .join("\n\n");

        let consensus_progress =
            others.iter().filter(|e| e.makes_progress).count() > others.len() / 2;

        format!(
            r#"You are reviewing evaluations from other committee members.

USER OBJECTIVE: {}

PROPOSED ACTION: {}

OTHER EVALUATIONS:
{}

CONSENSUS: {} agents think this makes progress

Consider their perspectives and either:
1. Maintain your position with stronger reasoning
2. Revise based on insights you missed

If most agents think this doesn't make progress, focus on:
- What specific aspect prevents forward movement?
- What alternative approach would make progress?

Provide your potentially revised evaluation in the same JSON format:
{{
    "makes_progress": true/false,
    "objective_alignment": 0.0-1.0,
    "implementation_quality": 0.0-1.0,
    "risk_assessment": 0.0-1.0,
    "reasoning": "explain your position after seeing other evaluations",
    "suggested_improvements": ["concrete suggestions based on committee discussion"]
}}"#,
            rubric.objective,
            action,
            other_evals,
            if consensus_progress { "Most" } else { "Few" }
        )
    }

    fn build_refine_prompt(
        &self,
        _state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
        others: &[AgentEvaluation],
        steering: &str,
    ) -> String {
        format!(
            r#"The committee has identified issues that need addressing.

USER OBJECTIVE: {}

PROPOSED ACTION: {}

COMMITTEE FEEDBACK:
{}

Key issues to address:
{}

Based on this feedback, provide a refined evaluation that:
1. Acknowledges the identified problems
2. Suggests how to modify the approach to make progress
3. Focuses on incremental improvement

Provide your refined evaluation in the same JSON format:
{{
    "makes_progress": true/false,
    "objective_alignment": 0.0-1.0,
    "implementation_quality": 0.0-1.0,
    "risk_assessment": 0.0-1.0,
    "reasoning": "explain how the feedback changes your assessment",
    "suggested_improvements": ["specific fixes for the identified issues"]
}}"#,
            rubric.objective,
            action,
            steering,
            others
                .iter()
                .filter(|e| !e.makes_progress)
                .flat_map(|e| e.suggested_improvements.iter())
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn build_final_prompt(
        &self,
        _state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
        _all_rounds: &[AgentEvaluation],
    ) -> String {
        format!(
            r#"This is the final evaluation round. Provide your definitive assessment.

OBJECTIVE: {}

PROPOSED ACTION: {}

Considering all previous discussions and refinements, provide your final scores.

Provide your evaluation in exactly this JSON format:
{{
    "latency_impact": <float between 0.0-2.0>,
    "memory_impact": <float between 0.0-2.0>,
    "relevance_impact": <float between 0.0-2.0>,
    "reasoning": "<final assessment incorporating all rounds of discussion>"
}}"#,
            rubric.objective, action
        )
    }

    fn build_evaluation_prompt(
        &self,
        state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
    ) -> String {
        format!(
            r#"You are an expert {} evaluator on an optimization committee.

USER OBJECTIVE: {}

CURRENT CODE:
```rust
{}
```

PROPOSED ACTION: {}

Your task is to evaluate whether this action makes incremental progress toward the user objective.

Consider from your {} perspective:
1. Does this action move us closer to the objective? (even small steps count)
2. How well does it align with what the user wants?
3. Is it well-implemented or just a hack?
4. What are the risks?

Provide your evaluation in exactly this JSON format:
{{
    "makes_progress": true/false,
    "objective_alignment": 0.0-1.0,
    "implementation_quality": 0.0-1.0,
    "risk_assessment": 0.0-1.0,
    "reasoning": "detailed explanation of your assessment",
    "suggested_improvements": ["improvement 1", "improvement 2", ...]
}}

Key guidelines:
- makes_progress: true if ANY incremental progress toward objective, false only if it moves away or adds no value
- objective_alignment: 1.0 = perfectly aligned, 0.0 = completely misaligned
- implementation_quality: 1.0 = production ready, 0.0 = broken
- risk_assessment: 1.0 = very safe, 0.0 = likely to break things
- Be generous with "makes_progress" - we want forward momentum
- Suggest concrete improvements even for good solutions"#,
            self.perspective, rubric.objective, state.code, action, self.perspective
        )
    }

    fn parse_evaluation_response(
        &self,
        response: &str,
        action: &str,
    ) -> Result<AgentEvaluation, CognitiveError> {
        // Try to extract JSON from response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];

        #[derive(Deserialize)]
        struct EvalResponse {
            makes_progress: bool,
            objective_alignment: f64,
            implementation_quality: f64,
            risk_assessment: f64,
            reasoning: String,
            #[serde(default)]
            suggested_improvements: Vec<String>}

        let eval: EvalResponse = serde_json::from_str(json_str).map_err(|e| {
            CognitiveError::ParseError(format!("Failed to parse evaluation: {}", e))
        })?;

        // Validate ranges
        if eval.objective_alignment < 0.0
            || eval.objective_alignment > 1.0
            || eval.implementation_quality < 0.0
            || eval.implementation_quality > 1.0
            || eval.risk_assessment < 0.0
            || eval.risk_assessment > 1.0
        {
            return Err(CognitiveError::ParseError(
                "Scores out of range [0.0, 1.0]".to_string(),
            ));
        }

        Ok(AgentEvaluation {
            agent_id: self.id.clone(),
            action: action.to_string(),
            makes_progress: eval.makes_progress,
            objective_alignment: eval.objective_alignment,
            implementation_quality: eval.implementation_quality,
            risk_assessment: eval.risk_assessment,
            reasoning: eval.reasoning,
            suggested_improvements: eval.suggested_improvements})
    }
}
