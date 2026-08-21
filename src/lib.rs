#![allow(dead_code)]

mod acp_client;
mod acp_state;
mod app;
mod auth_state;
mod command;
mod config;
mod diagnostics;
mod domain;
mod handlers;
mod highlight;
mod input;
mod markdown;
mod mesh;
mod mesh_state;
mod models_state;
mod navigation_state;
mod profiles_state;
mod protocol;
pub mod runtime;
mod server_manager;
mod session;
mod session_state;
mod slash;
mod theme;
mod themes_gen;
mod tool_detail;
mod ui;

pub(crate) use runtime::{ConnectionManagerEvent, ServerChannelMsg};
