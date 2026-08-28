use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::mesh_state::{MeshFocus, MeshInviteFormField, MeshState};
use crate::theme::Theme;
use crate::view_shared::truncate_with_ellipsis;

pub(crate) fn draw_mesh_popup(f: &mut Frame, mesh: &MeshState) {
    const MESH_POPUP_MIN_W: u16 = 54;
    const MESH_POPUP_MAX_W: u16 = 96;
    const MESH_POPUP_MIN_H: u16 = 16;
    const MESH_POPUP_MAX_H: u16 = 28;

    let area = f.area();
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(MESH_POPUP_MIN_W.min(area.width), MESH_POPUP_MAX_W);
    let popup_height = area
        .height
        .saturating_sub(2)
        .clamp(MESH_POPUP_MIN_H.min(area.height), MESH_POPUP_MAX_H);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(popup_width) / 2,
        y: area.y + area.height.saturating_sub(popup_height) / 3,
        width: popup_width,
        height: popup_height,
    };

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 2,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(4),
        height: popup_area.height.saturating_sub(2),
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(1),
        ])
        .split(inner);
    let title = mesh
        .mesh_status
        .as_ref()
        .map(|status| {
            format!(
                "mesh - {} peer{}",
                status.known_peer_count,
                if status.known_peer_count == 1 {
                    ""
                } else {
                    "s"
                }
            )
        })
        .unwrap_or_else(|| "mesh".to_string());
    let subtitle = mesh
        .selected_mesh_node_label()
        .map(|label| format!("selected node: {label}"))
        .unwrap_or_else(|| "no mesh nodes loaded".to_string());
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(title, Theme::popup_title())),
            Line::from(Span::styled(subtitle, Theme::status())),
        ])
        .style(Theme::popup_bg()),
        chunks[0],
    );

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(chunks[1]);

    let node_items: Vec<ListItem> = if mesh.mesh_nodes.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  no nodes",
            Theme::status(),
        )))]
    } else {
        mesh.mesh_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let selected = i == mesh.mesh_node_cursor;
                let style = if selected {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                let marker = if selected { "▸" } else { " " };
                let label =
                    truncate_with_ellipsis(&node.label, body[0].width.saturating_sub(14) as usize);
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {marker} "), style),
                    Span::styled(label, style),
                    Span::styled(format!("  {}", node.active_sessions), Theme::status()),
                ]))
            })
            .collect()
    };
    let node_block = Block::bordered()
        .title(" nodes ")
        .border_style(if matches!(mesh.mesh_focus, MeshFocus::Nodes) {
            Theme::status_accent()
        } else {
            Theme::status()
        })
        .style(Theme::popup_bg());
    let mut node_state = ListState::default().with_selected(
        (!mesh.mesh_nodes.is_empty()).then(|| mesh.mesh_node_cursor.min(mesh.mesh_nodes.len() - 1)),
    );
    f.render_stateful_widget(
        List::new(node_items).block(node_block),
        body[0],
        &mut node_state,
    );

    let sessions = mesh.selected_remote_sessions();
    let session_items: Vec<ListItem> = if sessions.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  no remote sessions",
            Theme::status(),
        )))]
    } else {
        sessions
            .iter()
            .enumerate()
            .map(|(i, session)| {
                let selected = i == mesh.remote_session_cursor;
                let style = if selected {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                let marker = if selected { "▸" } else { " " };
                let title = session.title.as_deref().unwrap_or(&session.id);
                let cwd = session.cwd.as_deref().unwrap_or("");
                let available = body[1].width.saturating_sub(7) as usize;
                let text = truncate_with_ellipsis(&format!("{title}  {cwd}"), available);
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {marker} "), style),
                    Span::styled(text, style),
                ]))
            })
            .collect()
    };
    let session_block = Block::bordered()
        .title(" remote sessions ")
        .border_style(if matches!(mesh.mesh_focus, MeshFocus::Sessions) {
            Theme::status_accent()
        } else {
            Theme::status()
        })
        .style(Theme::popup_bg());
    let mut session_state = ListState::default().with_selected(
        (!sessions.is_empty()).then(|| mesh.remote_session_cursor.min(sessions.len() - 1)),
    );
    f.render_stateful_widget(
        List::new(session_items).block(session_block),
        body[1],
        &mut session_state,
    );

    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("close/back  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("open/attach/create  ", Theme::status()),
        Span::styled("n ", Theme::status_accent()),
        Span::styled("new remote  ", Theme::status()),
        Span::styled("r ", Theme::status_accent()),
        Span::styled("refresh", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[2]);
}

pub(crate) fn draw_mesh_invite_popup(f: &mut Frame, mesh: &MeshState) {
    let area = f.area();
    let popup_width = area.width.saturating_sub(6).clamp(48.min(area.width), 54);
    let popup_height = area.height.saturating_sub(4).clamp(9.min(area.height), 11);
    let popup_area = sized_centered_rect(area, popup_width, popup_height);

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 2,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(4),
        height: popup_area.height.saturating_sub(2),
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Create Invite",
            Theme::popup_title(),
        )))
        .style(Theme::popup_bg()),
        chunks[0],
    );
    draw_mesh_invite_form(f, mesh, chunks[2]);

    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("close  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("create", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[3]);
}

pub(crate) fn draw_mesh_invite_qr_popup(f: &mut Frame, mesh: &MeshState) {
    let area = f.area();
    let (qr_width, qr_height) = mesh
        .mesh_invite
        .as_ref()
        .and_then(|invite| invite.qr_code.as_deref())
        .map(qr_text_size)
        .unwrap_or((36, 1));
    let title_width = "QR Code Invite".len() as u16;
    let hint_width = 32;
    let popup_width = qr_width
        .max(title_width)
        .max(hint_width)
        .saturating_add(5)
        .min(area.width.saturating_sub(2).max(1));
    let popup_height = qr_height
        .saturating_add(5)
        .min(area.height.saturating_sub(2).max(1));
    let popup_area = sized_centered_rect(area, popup_width, popup_height);

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 1,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(2),
        height: popup_area.height.saturating_sub(2),
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "QR Code Invite",
            Theme::popup_title(),
        )))
        .style(Theme::popup_bg()),
        chunks[0],
    );
    draw_mesh_invite_details(f, mesh, chunks[2]);

    if let Some(ref url) = mesh.mesh_clipboard_fallback {
        draw_mesh_clipboard_fallback(f, inner, url);
    }

    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("back  ", Theme::status()),
        Span::styled("u ", Theme::status_accent()),
        Span::styled("show URL  ", Theme::status()),
        Span::styled("C-y ", Theme::status_accent()),
        Span::styled("copy", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[3]);
}

fn qr_text_size(qr: &str) -> (u16, u16) {
    let width = qr
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(1) as u16;
    let height = qr.lines().count().max(1) as u16;
    (width, height)
}

fn sized_centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 3,
        width,
        height,
    }
}

fn draw_mesh_invite_form(f: &mut Frame, mesh: &MeshState, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    let field = mesh.mesh_invite_form_field;
    let line = |label: &'static str, value: &str, selected: bool| {
        let label_style = Theme::status();
        let value_style = if selected {
            Theme::selected()
        } else {
            Theme::popup_bg()
        };
        let cursor = if selected { ">" } else { " " };
        Line::from(vec![
            Span::styled(format!("{cursor} {label:<10}"), label_style),
            Span::styled(" ", Theme::popup_bg()),
            Span::styled(format!("{value:<18}"), value_style),
        ])
    };
    f.render_widget(
        Paragraph::new(line(
            "mesh name",
            if mesh.mesh_invite_name.is_empty() {
                "(none)"
            } else {
                &mesh.mesh_invite_name
            },
            matches!(field, MeshInviteFormField::MeshName),
        ))
        .style(Theme::popup_bg()),
        rows[0],
    );
    f.render_widget(
        Paragraph::new(line(
            "ttl",
            &mesh.mesh_invite_ttl,
            matches!(field, MeshInviteFormField::Ttl),
        ))
        .style(Theme::popup_bg()),
        rows[1],
    );
    f.render_widget(
        Paragraph::new(line(
            "max uses",
            &mesh.mesh_invite_max_uses,
            matches!(field, MeshInviteFormField::MaxUses),
        ))
        .style(Theme::popup_bg()),
        rows[2],
    );
    if let Some(message) = mesh.current_error() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" ! {message}"),
                Theme::error_text(),
            )))
            .style(Theme::popup_bg()),
            rows[4],
        );
    } else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " defaults: ttl 24h, max uses 1",
                Theme::status(),
            )))
            .style(Theme::popup_bg()),
            rows[4],
        );
    }
}

fn draw_mesh_invite_details(f: &mut Frame, mesh: &MeshState, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if let Some(invite) = mesh.mesh_invite.as_ref() {
        if let Some(qr) = invite.qr_code.as_deref() {
            for row in qr.lines() {
                lines.push(Line::from(Span::styled(
                    format!(" {row}"),
                    Theme::popup_bg(),
                )));
            }
        } else {
            lines.push(Line::from(Span::styled(
                " no QR code returned",
                Theme::status(),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            " create an invite to show the QR code",
            Theme::status(),
        )));
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

fn draw_mesh_clipboard_fallback(f: &mut Frame, area: Rect, url: &str) {
    let popup = Rect {
        x: area.x + 2,
        y: area.y + 2,
        width: area.width.saturating_sub(4),
        height: area.height.saturating_sub(4).max(6),
    };
    f.render_widget(Clear, popup);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup);
    let mut lines = vec![
        Line::from(Span::styled(
            " Clipboard not available",
            Theme::popup_title(),
        )),
        Line::from(Span::styled(
            " Copy this invite URL manually:",
            Theme::status(),
        )),
        Line::from(""),
    ];
    for chunk in wrap_plain_text(url, popup.width.saturating_sub(2) as usize) {
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
    f.render_widget(Paragraph::new(lines).style(Theme::popup_bg()), popup);
}

fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut remaining = text;
    let mut out = Vec::new();
    while !remaining.is_empty() {
        let take = remaining.len().min(width);
        out.push(remaining[..take].to_string());
        remaining = &remaining[take..];
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}
