use serde::de::DeserializeOwned;

use super::types::{DeserializerState, JsonPathDeserializer};

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
    /// Transition to navigating state
    #[inline]
    pub(super) fn transition_to_navigating(&mut self) {
        self.state = DeserializerState::Navigating;
    }

    /// Transition to processing array state  
    #[inline]
    pub(super) fn transition_to_processing_array(&mut self) {
        self.state = DeserializerState::ProcessingArray;
    }

    /// Transition to processing object state
    #[inline]
    pub(super) fn transition_to_processing_object(&mut self) {
        self.state = DeserializerState::ProcessingObject;
    }

    /// Transition to complete state
    #[inline]
    pub(super) fn transition_to_complete(&mut self) {
        self.state = DeserializerState::Complete;
    }
}
