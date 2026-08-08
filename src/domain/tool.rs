#[derive(Debug, Clone)]
pub struct DiffPreviewSection {
    pub header: String,
    pub old: String,
    pub new: String,
    pub start_line: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ShellOutputTail {
    pub lines: Vec<String>,
    pub hidden_line_count: usize,
}

#[derive(Debug, Clone)]
pub enum ToolDetail {
    None,
    /// Compact one-liner info for display after the tool name.
    Summary(String),
    /// One-liner header with output displayed below it.
    SummaryWithOutput {
        header: String,
        output: String,
    },
    Edit {
        file: String,
        old: String,
        new: String,
        start_line: Option<usize>,
    },
    MultiEdit {
        file: String,
        edit_count: usize,
        sections: Vec<DiffPreviewSection>,
    },
    ReplaceSymbol {
        title: String,
        sections: Vec<DiffPreviewSection>,
    },
    Shell {
        command: String,
        workdir: Option<String>,
        output_tail: Option<ShellOutputTail>,
    },
    ReadTool {
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    },
    WriteFile {
        path: String,
        content: String,
    },
}
