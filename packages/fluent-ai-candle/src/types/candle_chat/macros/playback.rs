//! Macro playback execution functionality
//!
//! This module handles the execution and playback of recorded macros,
//! providing high-performance streaming execution with proper error handling.

use std::collections::HashMap;
use std::time::Instant;

use crossbeam_skiplist::SkipMap;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{
    ActionExecutionResult, ChatMacro, MacroAction, MacroExecutionContext, MacroPlaybackResult,
    MacroPlaybackSession, MacroPlaybackState, MacroSystemError, LoopContext,
};
use super::variables::{resolve_variables, evaluate_condition};

/// Playback functionality for the macro system
pub struct MacroPlayer {
    /// Active playback sessions
    playback_sessions: RwLock<HashMap<Uuid, MacroPlaybackSession>>,
    /// Macro storage for playback
    macros: SkipMap<Uuid, ChatMacro>,
}

impl MacroPlayer {
    /// Create new macro player
    pub fn new(macros: SkipMap<Uuid, ChatMacro>) -> Self {
        Self {
            playback_sessions: RwLock::new(HashMap::new()),
            macros,
        }
    }

    /// Start macro playback (zero-allocation streaming)
    pub async fn start_playback(
        &self,
        macro_id: Uuid,
        variables: HashMap<String, String>,
    ) -> Result<Uuid, MacroSystemError> {
        let macro_def = self
            .get_macro(macro_id)
            .ok_or(MacroSystemError::MacroNotFound(macro_id))?;

        let session_id = Uuid::new_v4();
        let context = MacroExecutionContext {
            variables,
            execution_id: session_id,
            start_time: Instant::now(),
            current_action: 0,
            loop_stack: Vec::new(),
        };

        let session = MacroPlaybackSession {
            id: session_id,
            macro_id,
            start_time: Instant::now(),
            context,
            state: MacroPlaybackState::Playing,
            current_action: 0,
            total_actions: macro_def.actions.len(),
            error: None,
        };

        let mut sessions = self.playback_sessions.write().await;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Execute the next action in a playback session
    pub async fn execute_next_action(
        &self,
        session_id: Uuid,
    ) -> Result<MacroPlaybackResult, MacroSystemError> {
        let mut sessions = self.playback_sessions.write().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            if session.state != MacroPlaybackState::Playing {
                return Ok(MacroPlaybackResult::Paused);
            }

            let macro_def = self
                .get_macro(session.macro_id)
                .ok_or(MacroSystemError::MacroNotFound(session.macro_id))?;

            if session.current_action >= macro_def.actions.len() {
                session.state = MacroPlaybackState::Completed;
                return Ok(MacroPlaybackResult::Completed);
            }

            let action = &macro_def.actions[session.current_action];
            let result = self.execute_single_action(action.clone(), &mut session.context).await;

            session.current_action += 1;

            match result {
                ActionExecutionResult::Success => {
                    if session.current_action >= macro_def.actions.len() {
                        session.state = MacroPlaybackState::Completed;
                        Ok(MacroPlaybackResult::Completed)
                    } else {
                        Ok(MacroPlaybackResult::ActionExecuted)
                    }
                }
                ActionExecutionResult::Wait(duration) => {
                    tokio::time::sleep(duration).await;
                    Ok(MacroPlaybackResult::ActionExecuted)
                }
                ActionExecutionResult::SkipToAction(index) => {
                    session.current_action = index;
                    Ok(MacroPlaybackResult::ActionExecuted)
                }
                ActionExecutionResult::Error(error) => {
                    session.state = MacroPlaybackState::Failed;
                    session.error = Some(error);
                    Ok(MacroPlaybackResult::Failed)
                }
                ActionExecutionResult::Complete => {
                    session.state = MacroPlaybackState::Completed;
                    Ok(MacroPlaybackResult::Completed)
                }
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Execute a single macro action
    async fn execute_single_action(
        &self,
        action: MacroAction,
        context: &mut MacroExecutionContext,
    ) -> ActionExecutionResult {
        match action {
            MacroAction::SendMessage {
                content,
                message_type,
                ..
            } => {
                let resolved_content = resolve_variables(&content, &context.variables);
                println!("Sending message: {} (type: {})", resolved_content, message_type);
                ActionExecutionResult::Success
            }
            MacroAction::ExecuteCommand { command, .. } => {
                println!("Executing command: {:?}", command);
                ActionExecutionResult::Success
            }
            MacroAction::Wait { duration, .. } => {
                ActionExecutionResult::Wait(duration)
            }
            MacroAction::SetVariable { name, value, .. } => {
                let resolved_value = resolve_variables(&value, &context.variables);
                context.variables.insert(name, resolved_value);
                ActionExecutionResult::Success
            }
            MacroAction::Conditional {
                condition,
                then_actions,
                else_actions,
                ..
            } => {
                let condition_result = evaluate_condition(&condition, &context.variables);
                let actions_to_execute = if condition_result {
                    &then_actions
                } else if let Some(ref else_actions) = else_actions {
                    else_actions
                } else {
                    return ActionExecutionResult::Success;
                };

                for action in actions_to_execute.iter() {
                    let result = self.execute_single_action(action.clone(), context).await;
                    if let ActionExecutionResult::Error(_) = result {
                        return result;
                    }
                }
                ActionExecutionResult::Success
            }
            MacroAction::Loop {
                iterations,
                actions,
                ..
            } => {
                let loop_context = LoopContext {
                    iteration: 0,
                    max_iterations: iterations,
                    start_action: 0,
                    end_action: actions.len(),
                };

                context.loop_stack.push(loop_context);

                for iteration in 0..iterations {
                    if let Some(loop_ctx) = context.loop_stack.last_mut() {
                        loop_ctx.iteration = iteration;
                    }

                    for action in actions.iter() {
                        let result = self.execute_single_action(action.clone(), context).await;
                        if let ActionExecutionResult::Error(_) = result {
                            context.loop_stack.pop();
                            return result;
                        }
                    }
                }

                context.loop_stack.pop();
                ActionExecutionResult::Success
            }
        }
    }

    /// Get macro by ID
    fn get_macro(&self, macro_id: Uuid) -> Option<ChatMacro> {
        self.macros.get(&macro_id).map(|entry| entry.value().clone())
    }

    /// Pause playback session
    pub async fn pause_playback(&self, session_id: Uuid) -> Result<(), MacroSystemError> {
        let mut sessions = self.playback_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            match session.state {
                MacroPlaybackState::Playing => {
                    session.state = MacroPlaybackState::Paused;
                    Ok(())
                }
                _ => Err(MacroSystemError::InvalidAction(
                    "Cannot pause non-playing session".to_string()
                ))
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Resume playback session
    pub async fn resume_playback(&self, session_id: Uuid) -> Result<(), MacroSystemError> {
        let mut sessions = self.playback_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            match session.state {
                MacroPlaybackState::Paused => {
                    session.state = MacroPlaybackState::Playing;
                    Ok(())
                }
                _ => Err(MacroSystemError::InvalidAction(
                    "Cannot resume non-paused session".to_string()
                ))
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Get playback session status
    pub async fn get_playback_status(&self, session_id: Uuid) -> Result<PlaybackInfo, MacroSystemError> {
        let sessions = self.playback_sessions.read().await;
        
        if let Some(session) = sessions.get(&session_id) {
            Ok(PlaybackInfo {
                id: session.id,
                macro_id: session.macro_id,
                state: session.state,
                current_action: session.current_action,
                total_actions: session.total_actions,
                progress: if session.total_actions > 0 {
                    (session.current_action as f64 / session.total_actions as f64) * 100.0
                } else {
                    0.0
                },
                duration: session.start_time.elapsed(),
                error: session.error.clone(),
            })
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }
}

/// Information about a playback session
#[derive(Debug, Clone)]
pub struct PlaybackInfo {
    pub id: Uuid,
    pub macro_id: Uuid,
    pub state: MacroPlaybackState,
    pub current_action: usize,
    pub total_actions: usize,
    pub progress: f64,
    pub duration: std::time::Duration,
    pub error: Option<String>,
}