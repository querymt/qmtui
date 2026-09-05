use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::auth_state::{AuthPanel, AuthState};
use crate::domain::auth::{AuthMethod, AuthStatus, OAuthFlowKind, OAuthStatus};
use crate::theme::Theme;
use crate::view_shared::{
    ELLIPSIS, popup_rect, scroll_input, scroll_input_chars, wrap_display_width,
};

const AUTH_POPUP_MAX_W: u16 = 68;
const AUTH_POPUP_MIN_W: u16 = 44;

fn auth_method_label(method: AuthMethod) -> &'static str {
    match method {
        AuthMethod::OAuth => "OAuth",
        AuthMethod::ApiKey => "API Key",
        AuthMethod::EnvVar => "Env",
    }
}

fn auth_status_label(status: AuthStatus) -> &'static str {
    match status {
        AuthStatus::Unconfigurable => "OAuth required",
        AuthStatus::Expired => "Expired",
        AuthStatus::Active(method) => auth_method_label(method),
        AuthStatus::NotConfigured => "Not configured",
    }
}

pub(crate) fn draw_auth_popup(f: &mut Frame, auth: &AuthState) {
    let area = f.area();
    let desired_width = area.width.saturating_sub(4) as usize;
    let desired_height = (area.height as usize).saturating_mul(30) / 100;
    let popup_area = popup_rect(
        area,
        desired_width,
        desired_height,
        AUTH_POPUP_MIN_W as usize..=AUTH_POPUP_MAX_W as usize,
        15..=area.height as usize,
        2,
    );

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 1,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(2),
        height: popup_area.height.saturating_sub(2),
    };

    // If clipboard fallback is active, draw a simple URL display popup over everything.
    if let Some(ref url) = auth.clipboard_fallback {
        draw_auth_clipboard_fallback(f, inner, url);
        return;
    }

    // Determine detail area height based on panel state.
    let detail_height: u16 = match auth.panel {
        AuthPanel::List => {
            if auth.selected.is_some() {
                3 // status line + info
            } else {
                0
            }
        }
        AuthPanel::ApiKeyInput => 5,
        AuthPanel::OAuthFlow => 7,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),             // title
            Constraint::Length(1),             // filter
            Constraint::Length(1),             // spacer
            Constraint::Min(1),                // provider list
            Constraint::Length(detail_height), // detail panel
            Constraint::Length(1),             // hint
        ])
        .split(inner);

    // title
    f.render_widget(
        Paragraph::new(Span::styled("provider auth", Theme::popup_title()))
            .style(Theme::popup_bg()),
        chunks[0],
    );

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (auth_filter_display, auth_filter_cur) =
        scroll_input(&auth.filter, auth.filter.len(), avail);
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(auth_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    if auth.panel == AuthPanel::List && chunks[1].width > 2 && chunks[1].height > 0 {
        f.set_cursor_position((
            chunks[1]
                .x
                .saturating_add(2)
                .saturating_add(auth_filter_cur as u16),
            chunks[1].y,
        ));
    }

    // provider list
    let filtered = auth.filtered_providers();
    let list_w = chunks[3].width as usize;

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .map(|(i, (_, provider))| {
            let selected = i == auth.cursor;
            let status = provider.auth_status();
            let badge = auth_status_label(status);
            let badge_active = provider.is_auth_active();
            let is_expired = provider.oauth_status == Some(OAuthStatus::Expired);

            // Badge styling
            let badge_style = if selected {
                Theme::selected()
            } else if is_expired {
                ratatui::style::Style::default()
                    .fg(Theme::warn())
                    .bg(Theme::bg_dim())
            } else if badge_active {
                ratatui::style::Style::default()
                    .fg(Theme::ok())
                    .bg(Theme::bg_dim())
            } else {
                Theme::status()
            };

            let name_style = if selected {
                Theme::selected()
            } else {
                Theme::popup_bg()
            };

            let badge_str = format!("[{badge}]");
            let badge_len = badge_str.chars().count();
            let name = &provider.display_name;
            let avail = list_w.saturating_sub(badge_len + 3);
            let name_display = if name.chars().count() > avail {
                let t: String = name.chars().take(avail.saturating_sub(1)).collect();
                format!("{t}{ELLIPSIS}")
            } else {
                name.to_string()
            };
            let gap = avail.saturating_sub(name_display.chars().count());

            ListItem::new(Line::from(vec![
                Span::styled(format!(" {name_display}"), name_style),
                Span::styled(" ".repeat(gap), name_style),
                Span::styled(format!(" {badge_str} "), badge_style),
            ]))
        })
        .collect();

    if items.is_empty() {
        let msg = if auth.filter.is_empty() {
            "loading providers..."
        } else {
            "no providers match filter"
        };
        f.render_widget(
            Paragraph::new(Span::styled(format!(" {msg}"), Theme::status()))
                .style(Theme::popup_bg()),
            chunks[3],
        );
    } else {
        let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
        let visible_rows = chunks[3].height as usize;
        let offset = auth.cursor.saturating_sub(visible_rows.saturating_sub(1));
        let mut state = ListState::default()
            .with_offset(offset)
            .with_selected(Some(auth.cursor));
        f.render_stateful_widget(list, chunks[3], &mut state);
    }

    // detail panel
    if detail_height > 0 {
        draw_auth_detail_panel(f, auth, chunks[4]);
    }

    // hint
    let hint = match auth.panel {
        AuthPanel::List => {
            let mut spans = vec![
                Span::styled(" esc ", Theme::status_accent()),
                Span::styled("close  ", Theme::status()),
                Span::styled("enter ", Theme::status_accent()),
                Span::styled("select  ", Theme::status()),
            ];
            // Show C-d contextually when a provider with clearable credentials is selected
            if let Some(idx) = auth.selected
                && let Some(provider) = auth.providers.get(idx)
            {
                if provider.oauth_status == Some(OAuthStatus::Connected) {
                    spans.push(Span::styled("C-d ", Theme::status_accent()));
                    spans.push(Span::styled("disconnect  ", Theme::status()));
                } else if provider.has_stored_api_key {
                    spans.push(Span::styled("C-d ", Theme::status_accent()));
                    spans.push(Span::styled("clear key  ", Theme::status()));
                }
            }
            spans.push(Span::styled("C-k ", Theme::status_accent()));
            spans.push(Span::styled("api key  ", Theme::status()));
            spans.push(Span::styled("C-o ", Theme::status_accent()));
            spans.push(Span::styled("oauth", Theme::status()));
            Line::from(spans)
        }
        AuthPanel::ApiKeyInput => Line::from(vec![
            Span::styled(" esc ", Theme::status_accent()),
            Span::styled("back  ", Theme::status()),
            Span::styled("enter ", Theme::status_accent()),
            Span::styled("save  ", Theme::status()),
            Span::styled("tab ", Theme::status_accent()),
            Span::styled("toggle mask  ", Theme::status()),
            Span::styled("C-d ", Theme::status_accent()),
            Span::styled("clear key", Theme::status()),
        ]),
        AuthPanel::OAuthFlow => Line::from(vec![
            Span::styled(" esc ", Theme::status_accent()),
            Span::styled("back  ", Theme::status()),
            Span::styled("C-y ", Theme::status_accent()),
            Span::styled("copy url  ", Theme::status()),
            Span::styled("enter ", Theme::status_accent()),
            Span::styled("complete", Theme::status()),
        ]),
    };
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[5]);
}

fn draw_auth_detail_panel(f: &mut Frame, auth: &AuthState, area: Rect) {
    let Some(idx) = auth.selected else {
        return;
    };
    let Some(provider) = auth.providers.get(idx) else {
        return;
    };

    match auth.panel {
        AuthPanel::List => {
            // Show selected provider status summary
            let mut lines: Vec<Line<'static>> = Vec::new();

            let status_label = auth_status_label(provider.auth_status());
            let is_active = provider.is_auth_active();
            let status_style = if is_active {
                ratatui::style::Style::default()
                    .fg(Theme::ok())
                    .bg(Theme::bg_dim())
            } else {
                Theme::status()
            };

            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", provider.display_name), Theme::popup_title()),
                Span::styled(format!("[{status_label}]"), status_style),
            ]));

            if provider.is_unconfigurable() {
                lines.push(Line::from(Span::styled(
                    " OAuth required (not available in this build)",
                    Theme::status(),
                )));
            } else if let Some(ref env_var) = provider.env_var_name {
                lines.push(Line::from(Span::styled(
                    format!(" env: {env_var}"),
                    Theme::status(),
                )));
            }

            if let Some((success, message)) = auth.feedback_for_provider(&provider.provider) {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {message}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }
        }

        AuthPanel::ApiKeyInput => {
            let mut lines: Vec<Line<'static>> = Vec::new();

            lines.push(Line::from(vec![
                Span::styled(
                    format!(" API Key for {} ", provider.display_name),
                    Theme::popup_title(),
                ),
                if let Some(ref env_var) = provider.env_var_name {
                    Span::styled(format!(" {env_var}"), Theme::status())
                } else {
                    Span::raw("")
                },
            ]));

            if provider.has_stored_api_key {
                lines.push(Line::from(Span::styled(
                    " key stored in keychain",
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim()),
                )));
            }

            // Input line
            let cursor_chars = auth.api_key_input[..auth.api_key_cursor].chars().count();
            let display_input = if auth.api_key_masked && !auth.api_key_input.is_empty() {
                "\u{2022}".repeat(auth.api_key_input.chars().count())
            } else {
                auth.api_key_input.clone()
            };
            let placeholder = if provider.has_stored_api_key {
                "new key to update..."
            } else {
                "enter API key..."
            };
            // " > " prefix = 3 cols
            let avail = area.width.saturating_sub(3) as usize;
            let (input_text, api_key_cur) = if auth.api_key_input.is_empty() {
                (placeholder.to_string(), 0usize)
            } else {
                let (vis, col) = scroll_input_chars(&display_input, cursor_chars, avail);
                (vis, col)
            };
            let input_style = if auth.api_key_input.is_empty() {
                Theme::status()
            } else {
                Theme::popup_bg()
            };
            lines.push(Line::from(vec![
                Span::styled(" > ", Theme::popup_title()),
                Span::styled(input_text, input_style),
            ]));

            if let Some((success, message)) = auth.feedback_for_provider(&provider.provider) {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {message}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }

            // Position cursor in the input field
            let input_row_idx = if provider.has_stored_api_key { 2 } else { 1 };
            if area.width > 3 && (input_row_idx as u16) < area.height {
                f.set_cursor_position((
                    area.x.saturating_add(3).saturating_add(api_key_cur as u16),
                    area.y.saturating_add(input_row_idx as u16),
                ));
            }
        }

        AuthPanel::OAuthFlow => {
            let mut lines: Vec<Line<'static>> = Vec::new();

            if let Some(ref flow) = auth.oauth_flow {
                let is_device_poll = flow.flow_kind == OAuthFlowKind::DevicePoll;

                lines.push(Line::from(Span::styled(
                    format!(" OAuth for {}", flow.provider),
                    Theme::popup_title(),
                )));

                // Truncate URL for display
                let url = &flow.authorization_url;
                let avail = area.width.saturating_sub(3) as usize;
                let url_display = if url.chars().count() > avail {
                    let t: String = url.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    url.to_string()
                };
                lines.push(Line::from(Span::styled(
                    format!(" {url_display}"),
                    Theme::status(),
                )));

                if is_device_poll {
                    lines.push(Line::from(Span::styled(
                        " Open URL, approve, then press Enter to check",
                        Theme::status(),
                    )));
                } else {
                    // " > " prefix = 3 cols
                    let avail = area.width.saturating_sub(3) as usize;
                    let (response_display, _) =
                        scroll_input(&auth.oauth_response, auth.oauth_response_cursor, avail);
                    lines.push(Line::from(Span::styled(
                        " Open URL, approve, paste callback below:",
                        Theme::status(),
                    )));
                    lines.push(Line::from(vec![
                        Span::styled(" > ", Theme::popup_title()),
                        Span::styled(response_display, Theme::popup_bg()),
                    ]));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    " Starting OAuth flow...",
                    Theme::status(),
                )));
            }

            if let Some((success, message)) = auth.feedback_for_provider(&provider.provider) {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {message}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }

            // Position cursor on callback input (non-device-poll only, row 3)
            if auth.oauth_flow.is_some()
                && auth
                    .oauth_flow
                    .as_ref()
                    .is_some_and(|f| f.flow_kind != OAuthFlowKind::DevicePoll)
            {
                let avail = area.width.saturating_sub(3) as usize;
                let (_, oauth_cur) =
                    scroll_input(&auth.oauth_response, auth.oauth_response_cursor, avail);
                if area.width > 3 && 3 < area.height {
                    f.set_cursor_position((
                        area.x.saturating_add(3).saturating_add(oauth_cur as u16),
                        area.y.saturating_add(3),
                    ));
                }
            }
        }
    }
}

fn draw_auth_clipboard_fallback(f: &mut Frame, area: Rect, url: &str) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        " Clipboard not available",
        Theme::popup_title(),
    )));
    lines.push(Line::from(Span::styled(
        " Copy this URL manually:",
        Theme::status(),
    )));
    lines.push(Line::from(""));

    let avail = area.width.saturating_sub(2) as usize;
    for chunk in wrap_display_width(url, avail) {
        lines.push(Line::from(Span::styled(
            format!(" {chunk}"),
            Theme::popup_bg(),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " press any key to dismiss",
        Theme::status(),
    )));

    for (i, line) in lines.into_iter().enumerate() {
        if i as u16 >= area.height {
            break;
        }
        let row = Rect {
            x: area.x,
            y: area.y + i as u16,
            width: area.width,
            height: 1,
        };
        f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::auth::AuthProviderEntry;

    fn auth_provider(display_name: &str, status: AuthStatus) -> AuthProviderEntry {
        let (supports_oauth, env_var_name) = match status {
            AuthStatus::Unconfigurable => (false, None),
            AuthStatus::Active(AuthMethod::OAuth) | AuthStatus::Expired => (true, None),
            AuthStatus::Active(AuthMethod::ApiKey) | AuthStatus::Active(AuthMethod::EnvVar) => (
                false,
                Some(format!("{}_API_KEY", display_name.to_uppercase())),
            ),
            AuthStatus::NotConfigured => (
                true,
                Some(format!("{}_API_KEY", display_name.to_uppercase())),
            ),
        };
        AuthProviderEntry {
            provider: display_name.to_lowercase().replace(' ', "-"),
            display_name: display_name.into(),
            oauth_status: match status {
                AuthStatus::Expired => Some(OAuthStatus::Expired),
                AuthStatus::Active(AuthMethod::OAuth) => Some(OAuthStatus::Connected),
                AuthStatus::Unconfigurable
                | AuthStatus::Active(AuthMethod::ApiKey)
                | AuthStatus::Active(AuthMethod::EnvVar) => None,
                AuthStatus::NotConfigured => Some(OAuthStatus::NotAuthenticated),
            },
            has_stored_api_key: status == AuthStatus::Active(AuthMethod::ApiKey),
            has_env_api_key: status == AuthStatus::Active(AuthMethod::EnvVar),
            env_var_name,
            supports_oauth,
            preferred_method: None,
        }
    }

    fn buffer_line(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect()
    }

    fn find_buffer_text(buffer: &ratatui::buffer::Buffer, needle: &str) -> Option<(u16, u16)> {
        for y in 0..buffer.area.height {
            let line = buffer_line(buffer, y);
            if let Some(byte_idx) = line.find(needle) {
                return Some((line[..byte_idx].chars().count() as u16, y));
            }
        }
        None
    }

    fn find_last_buffer_text(buffer: &ratatui::buffer::Buffer, needle: &str) -> Option<(u16, u16)> {
        for y in (0..buffer.area.height).rev() {
            let line = buffer_line(buffer, y);
            if let Some(byte_idx) = line.find(needle) {
                return Some((line[..byte_idx].chars().count() as u16, y));
            }
        }
        None
    }

    #[test]
    fn auth_labels_map_semantic_methods_and_statuses() {
        assert_eq!(auth_method_label(AuthMethod::OAuth), "OAuth");
        assert_eq!(auth_method_label(AuthMethod::ApiKey), "API Key");
        assert_eq!(auth_method_label(AuthMethod::EnvVar), "Env");
        assert_eq!(
            auth_status_label(AuthStatus::Unconfigurable),
            "OAuth required"
        );
        assert_eq!(auth_status_label(AuthStatus::Expired), "Expired");
        assert_eq!(
            auth_status_label(AuthStatus::Active(AuthMethod::OAuth)),
            "OAuth"
        );
        assert_eq!(
            auth_status_label(AuthStatus::Active(AuthMethod::ApiKey)),
            "API Key"
        );
        assert_eq!(
            auth_status_label(AuthStatus::Active(AuthMethod::EnvVar)),
            "Env"
        );
        assert_eq!(
            auth_status_label(AuthStatus::NotConfigured),
            "Not configured"
        );
    }

    #[test]
    fn auth_popup_renders_exact_badges_and_styles_in_list_and_detail() {
        let mut auth = AuthState::new();
        auth.providers = vec![
            auth_provider("OAuth Active", AuthStatus::Active(AuthMethod::OAuth)),
            auth_provider("API Active", AuthStatus::Active(AuthMethod::ApiKey)),
            auth_provider("Env Active", AuthStatus::Active(AuthMethod::EnvVar)),
            auth_provider("Expired Auth", AuthStatus::Expired),
            auth_provider("OAuth Missing", AuthStatus::Unconfigurable),
            auth_provider("No Auth", AuthStatus::NotConfigured),
        ];
        auth.cursor = auth.providers.len();

        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| draw_auth_popup(frame, &auth))
            .unwrap();
        let buffer = terminal.backend().buffer();

        for (name, label, foreground) in [
            ("OAuth Active", "[OAuth]", Theme::ok()),
            ("API Active", "[API Key]", Theme::ok()),
            ("Env Active", "[Env]", Theme::ok()),
            ("Expired Auth", "[Expired]", Theme::warn()),
        ] {
            let row = find_buffer_text(buffer, name).expect("provider row").1;
            let badge = find_buffer_text(buffer, label).expect("provider badge");
            assert_eq!(badge.1, row);
            assert_eq!(buffer[badge].fg, foreground);
        }
        for (name, label) in [
            ("OAuth Missing", "[OAuth required]"),
            ("No Auth", "[Not configured]"),
        ] {
            let row = find_buffer_text(buffer, name).expect("provider row").1;
            let badge = find_buffer_text(buffer, label).expect("provider badge");
            assert_eq!(badge.1, row);
            assert_eq!(buffer[badge].fg, Theme::status().fg.unwrap());
            assert_eq!(buffer[badge].bg, Theme::status().bg.unwrap());
        }

        for (index, label, active) in [
            (0, "[OAuth]", true),
            (1, "[API Key]", true),
            (2, "[Env]", true),
            (3, "[Expired]", false),
            (4, "[OAuth required]", false),
            (5, "[Not configured]", false),
        ] {
            auth.selected = Some(index);
            let backend = ratatui::backend::TestBackend::new(80, 24);
            let mut terminal = ratatui::Terminal::new(backend).unwrap();
            terminal
                .draw(|frame| draw_auth_popup(frame, &auth))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let badge = find_last_buffer_text(buffer, label).expect("detail badge");
            let (expected_fg, expected_bg) = if active {
                (Theme::ok(), Theme::bg_dim())
            } else {
                (Theme::status().fg.unwrap(), Theme::status().bg.unwrap())
            };
            assert_eq!(buffer[badge].fg, expected_fg);
            assert_eq!(buffer[badge].bg, expected_bg);
        }
    }
}
