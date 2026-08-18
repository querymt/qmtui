#![allow(dead_code)]

mod acp_client;
mod acp_state;
mod app;
mod command;
mod config;
mod domain;
mod handlers;
mod highlight;
mod input;
mod markdown;
mod mesh;
mod protocol;
pub mod runtime;
mod server_manager;
mod session;
mod slash;
mod theme;
mod themes_gen;
mod tool_detail;
mod ui;

pub(crate) use runtime::{ConnectionManagerEvent, ServerChannelMsg};
