use fluent_ai_candle::streaming::decoder::state::*;
use fluent_ai_candle::streaming::decoder::*;

#[test]
    fn test_state_transitions() {
        let mut state = DecoderState::ready();
        assert!(state.is_ready());

        state = DecoderState::partial(vec![0xC3]);
        assert!(state.is_partial());
        assert_eq!(state.pending_bytes(), Some(&[0xC3][..]));

        state = DecoderState::error("test error");
        assert!(state.is_error());
        assert_eq!(state.error_message(), Some("test error"));

        state.reset();
        assert!(state.is_ready());
    }

    #[test]
    fn test_transition() {
        let mut state = DecoderState::ready();
        
        // Successful transition
        let result: Result<&str, &str> = Ok("success");
        assert_eq!(state.transition(result), Some("success"));
        assert!(!state.is_error());
        
        // Error transition
        let result: Result<&str, &str> = Err("failure");
        assert_eq!(state.transition(result), None);
        assert!(state.is_error());
        
        // Reset on success after error
        let result: Result<&str, &str> = Ok("recovered");
        assert_eq!(state.transition(result), Some("recovered"));
        assert!(!state.is_error());
    }
