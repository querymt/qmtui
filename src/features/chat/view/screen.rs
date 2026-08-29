use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
};

use crate::chat_state::ChatState;
use crate::composer_state::ComposerState;
use crate::connection_state::ConnState;
use crate::delegates_state::DelegatesState;
use crate::models_state::ModelsState;
use crate::profiles_state::ProfilesState;
use crate::render_state::{RenderState, SessionIdentity, ThemeCacheKey};
use crate::session_state::SessionsState;
use crate::view_shared::{SpinnerKind, spinner};

use super::header::{HeaderRenderInput, draw_chat_header};
use super::viewport::{MessagesRenderInput, draw_messages};
use super::{
    completion_panel_height, draw_completion_panel, draw_elicitation_popup, draw_input_panel,
    elicitation_popup_height, input_layout_metrics,
};

pub(crate) struct ChatScreenInput<'a> {
    pub(crate) chat: &'a ChatState,
    pub(crate) composer: &'a ComposerState,
    pub(crate) sessions: &'a SessionsState,
    pub(crate) delegates: &'a DelegatesState,
    pub(crate) models: &'a ModelsState,
    pub(crate) profiles: &'a ProfilesState,
    pub(crate) effective_cwd: Option<String>,
    pub(crate) cwd_label: Option<String>,
    pub(crate) mesh_node_count: Option<u32>,
    pub(crate) chord: bool,
    pub(crate) connection: ConnState,
    pub(crate) theme: ThemeCacheKey,
}

pub(super) fn session_identity(sessions: &SessionsState) -> SessionIdentity {
    let session_id = sessions.session_id.clone();
    let remote_node_id = session_id
        .as_deref()
        .and_then(|id| sessions.session_remote_node_id(id))
        .map(str::to_string);
    let is_remote = session_id
        .as_deref()
        .is_some_and(|id| sessions.is_remote_session_id(id));
    SessionIdentity::new(session_id, remote_node_id, is_remote)
}

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn draw_header(frame: &mut Frame, area: ratatui::layout::Rect, input: &ChatScreenInput<'_>) {
    draw_chat_header(
        frame,
        area,
        HeaderRenderInput {
            chat: input.chat,
            delegates: input.delegates,
            sessions: input.sessions,
            models: input.models,
            profiles: input.profiles,
            cwd_label: input.cwd_label.as_deref(),
            mesh_node_count: input.mesh_node_count,
            chord: input.chord,
            connection: input.connection,
        },
    );
}

fn draw_message_viewport(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    input: &ChatScreenInput<'_>,
    render: &mut RenderState,
) {
    draw_messages(
        frame,
        area,
        MessagesRenderInput {
            session_identity: session_identity(input.sessions),
            chat: input.chat,
            delegates: input.delegates,
            effective_cwd: input.effective_cwd.clone(),
            theme: input.theme,
            now_unix_secs: now_unix_secs(),
        },
        render,
    );
}

pub(crate) fn draw_delegate_view(
    frame: &mut Frame,
    input: ChatScreenInput<'_>,
    render: &mut RenderState,
) {
    let area = frame.area();

    let (input_height, input_layout) = input_layout_metrics(input.composer, render, area);
    let elicitation_height = elicitation_popup_height(
        input.chat.elicitation.as_ref(),
        input.chat.elicitation_ui.as_ref(),
        render,
        area,
    );

    let chunks = if elicitation_height > 0 {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(elicitation_height),
                Constraint::Length(1),
                Constraint::Length(input_height),
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area)
    };

    draw_header(frame, chunks[0], &input);
    draw_message_viewport(frame, chunks[1], &input, render);

    if elicitation_height > 0 {
        if let (Some(state), Some(ui)) = (&input.chat.elicitation, &input.chat.elicitation_ui) {
            draw_elicitation_popup(frame, state, ui, render, chunks[2]);
        }
        let spinner_frame = spinner(SpinnerKind::Braille, render.tick);
        draw_input_panel(
            frame,
            input.chat,
            &input.sessions.agent_mode,
            spinner_frame,
            render,
            (chunks[3], chunks[4]),
            input_layout,
        );
    }
}

pub(crate) fn draw_chat(frame: &mut Frame, input: ChatScreenInput<'_>, render: &mut RenderState) {
    let area = frame.area();
    let completion_height = completion_panel_height(input.composer);

    let (input_height, input_layout) = input_layout_metrics(input.composer, render, area);
    let elicitation_height = elicitation_popup_height(
        input.chat.elicitation.as_ref(),
        input.chat.elicitation_ui.as_ref(),
        render,
        area,
    );

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(completion_height),
            Constraint::Length(elicitation_height),
            Constraint::Length(1),
            Constraint::Length(input_height),
        ])
        .split(area);

    draw_header(frame, chunks[0], &input);
    draw_message_viewport(frame, chunks[1], &input, render);

    if completion_height > 0 {
        draw_completion_panel(
            frame,
            input.composer,
            spinner(SpinnerKind::Braille, render.tick),
            chunks[2],
        );
    }

    if elicitation_height > 0
        && let (Some(state), Some(ui)) = (&input.chat.elicitation, &input.chat.elicitation_ui)
    {
        draw_elicitation_popup(frame, state, ui, render, chunks[3]);
    }

    let spinner_frame = spinner(SpinnerKind::Braille, render.tick);
    draw_input_panel(
        frame,
        input.chat,
        &input.sessions.agent_mode,
        spinner_frame,
        render,
        (chunks[4], chunks[5]),
        input_layout,
    );
}
