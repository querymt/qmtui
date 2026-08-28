use ratatui::{Frame, layout::Rect, style::Style, text::Span};

use crate::chat_state::ChatState;
use crate::connection_state::ConnState;
use crate::delegates_state::DelegatesState;
use crate::domain::activity::DelegateStatus;
use crate::models_state::ModelsState;
use crate::profiles_state::ProfilesState;
use crate::session_state::SessionsState;
use crate::theme::Theme;
use crate::view_shared::{draw_header, mesh_header_span};

const ICON_CONTEXT: &str = "\u{1F5AA}";
const ICON_TOOLS: &str = "\u{2692}";
pub(crate) const ICON_DELEGATES: &str = "\u{2387}";
pub(crate) const ICON_MULTI_SESSION: &str = "𐬽";

pub(super) struct HeaderRenderInput<'a> {
    pub(super) chat: &'a ChatState,
    pub(super) delegates: &'a DelegatesState,
    pub(super) sessions: &'a SessionsState,
    pub(super) models: &'a ModelsState,
    pub(super) profiles: &'a ProfilesState,
    pub(super) cwd_label: Option<&'a str>,
    pub(super) mesh_node_count: Option<u32>,
    pub(super) chord: bool,
    pub(super) connection: ConnState,
}

fn format_status_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!(
            "{}h{:02}m{:02}s",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    }
}

fn build_chat_header_spans(
    input: &HeaderRenderInput<'_>,
) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let model_str = match (&input.models.current_provider, &input.models.current_model) {
        (Some(provider), Some(model)) => {
            if let Some(node_id) = input.models.current_model_node_id.as_deref() {
                let label = input
                    .models
                    .models
                    .iter()
                    .find(|entry| {
                        ModelsState::model_entry_matches_node(entry, provider, model, Some(node_id))
                    })
                    .and_then(|entry| entry.node_label.as_deref())
                    .unwrap_or(node_id);
                format!("{provider}/{model}@{label}")
            } else {
                format!("{provider}/{model}")
            }
        }
        _ => "no model".into(),
    };
    let session_id = input
        .sessions
        .session_id
        .as_deref()
        .map(|session_id| {
            if session_id.len() > 8 {
                &session_id[..8]
            } else {
                session_id
            }
        })
        .unwrap_or("???");

    let mut right_spans = Vec::new();

    if let Some(duration) = input.chat.llm_request_elapsed() {
        right_spans.push(Span::styled(
            format!(" {} ", format_status_duration(duration)),
            Theme::status(),
        ));
    }

    if let Some(context_tokens) = input.chat.session_stats.latest_context_tokens
        && context_tokens > 0
        && input.chat.context_limit > 0
    {
        let percentage = (context_tokens as f64 / input.chat.context_limit as f64 * 100.0) as u32;
        right_spans.push(Span::styled(
            format!(" {ICON_CONTEXT} {percentage}% "),
            Theme::status(),
        ));
    }

    if input.chat.session_stats.total_tool_calls > 0 {
        right_spans.push(Span::styled(
            format!(
                " {ICON_TOOLS} {} ",
                input.chat.session_stats.total_tool_calls
            ),
            Theme::status(),
        ));
    }

    if let Some(cost) = input.chat.cumulative_cost
        && cost > 0.0
    {
        right_spans.push(Span::styled(
            format!(" ${cost:.4} "),
            Theme::status_accent(),
        ));
    }

    if !input.delegates.delegate_entries.is_empty() {
        let (mut done, mut has_failed, mut has_running, mut awaiting_input) =
            (0usize, false, false, false);
        for entry in &input.delegates.delegate_entries {
            match entry.status {
                DelegateStatus::Completed | DelegateStatus::Cancelled => done += 1,
                DelegateStatus::Failed => has_failed = true,
                DelegateStatus::InProgress => has_running = true,
            }
            awaiting_input |= entry.awaiting_input();
        }
        let total = input.delegates.delegate_entries.len();
        let style = if has_failed {
            Theme::error_on_dim()
        } else if has_running {
            Theme::status_accent()
        } else {
            Theme::status()
        };
        right_spans.push(Span::styled(
            format!(" {ICON_DELEGATES} {done}/{total} "),
            style,
        ));
        if awaiting_input {
            right_spans.push(Span::styled(" awaiting input ", Theme::mode_badge("plan")));
        }
    }

    let other_active_session_count = input.sessions.other_active_session_count();
    if other_active_session_count > 0 {
        right_spans.push(Span::styled(
            format!(" {ICON_MULTI_SESSION} {other_active_session_count} "),
            Theme::status(),
        ));
    }

    if let Some(span) = mesh_header_span(input.mesh_node_count) {
        right_spans.push(span);
    }

    let effort_label = input.models.reasoning_effort_label();
    right_spans.push(Span::styled(
        format!(" {model_str}"),
        Theme::status_accent(),
    ));
    right_spans.push(Span::styled(":", Theme::reasoning_effort_sep()));
    right_spans.push(Span::styled(
        format!("{effort_label} "),
        Theme::reasoning_effort_level(),
    ));

    let mut left_spans = vec![Span::styled(format!(" {session_id}"), Theme::status())];
    if let Some(cwd_label) = input.cwd_label {
        left_spans.push(Span::styled(
            format!(":{cwd_label}"),
            Style::default().fg(Theme::ok()).bg(Theme::bg_dim()),
        ));
    }
    left_spans.push(Span::styled(
        format!(" {} ", input.sessions.agent_mode),
        Theme::mode_badge(&input.sessions.agent_mode),
    ));
    if input.delegates.parent_session_id.is_some() {
        left_spans.push(Span::styled(" \u{2b11} child ", Theme::status_accent()));
    }

    let current_profile_id = input
        .sessions
        .session_id
        .as_deref()
        .and_then(|session_id| input.profiles.session_profile_id(session_id));
    let profile_label = current_profile_id
        .map(|profile_id| input.profiles.profile_display_name(profile_id))
        .unwrap_or_else(|| {
            if input.sessions.current_session_is_remote() {
                "remote".to_string()
            } else {
                input.profiles.active_profile_label()
            }
        });
    let active_profile_id = input.profiles.active_profile_id.as_deref();
    let profile_text = if current_profile_id.is_some()
        && active_profile_id.is_some()
        && current_profile_id != active_profile_id
    {
        format!(
            " profile:{} (new:{}) ",
            profile_label,
            input.profiles.active_profile_label()
        )
    } else {
        format!(" profile:{profile_label} ")
    };
    left_spans.push(Span::styled(profile_text, Theme::status()));

    (left_spans, right_spans)
}

pub(super) fn draw_chat_header(frame: &mut Frame, area: Rect, input: HeaderRenderInput<'_>) {
    let (left, right) = build_chat_header_spans(&input);
    draw_header(frame, area, left, right, input.chord, input.connection);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::profile::ProfileInfo;

    #[test]
    fn chat_header_shows_current_and_distinct_new_profile_labels() {
        let chat = ChatState::new();
        let delegates = DelegatesState::new();
        let mut sessions = SessionsState::new();
        let models = ModelsState::new();
        let mut profiles = ProfilesState::new();
        sessions.session_id = Some("session-1".into());
        profiles.profiles = vec![
            ProfileInfo {
                id: "current".into(),
                name: "Current".into(),
                ..Default::default()
            },
            ProfileInfo {
                id: "next".into(),
                name: "Next".into(),
                ..Default::default()
            },
        ];
        profiles.active_profile_id = Some("next".into());
        profiles.bind_session_profile("session-1".into(), "current".into());

        let (left, _) = build_chat_header_spans(&HeaderRenderInput {
            chat: &chat,
            delegates: &delegates,
            sessions: &sessions,
            models: &models,
            profiles: &profiles,
            cwd_label: None,
            mesh_node_count: None,
            chord: false,
            connection: ConnState::Connected,
        });
        let text = left
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(text.contains("profile:Current (new:Next)"));
    }
}
