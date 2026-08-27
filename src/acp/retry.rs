use std::time::Duration;

pub(crate) fn websocket_delay(attempt: u32) -> Duration {
    Duration::from_millis(250 * (1u64 << attempt.min(5)))
}

#[cfg(test)]
mod tests {
    use super::websocket_delay;

    #[test]
    fn websocket_backoff_caps_at_eight_seconds() {
        let delays = (0..=8)
            .map(|attempt| websocket_delay(attempt).as_millis())
            .collect::<Vec<_>>();
        assert_eq!(delays, [250, 500, 1000, 2000, 4000, 8000, 8000, 8000, 8000]);
    }
}
