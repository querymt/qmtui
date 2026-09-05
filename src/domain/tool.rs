#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellOutput {
    pub stdout: String,
    pub stderr: String,
    /// Number of nonblank output lines omitted before this captured output.
    pub preceding_line_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiEditSection {
    pub edit_index: usize,
    pub replace_all: bool,
    pub old: String,
    pub new: String,
    pub start_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolReplacement {
    pub path: String,
    pub symbol: String,
    pub new_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolDiffSection {
    pub path: String,
    pub symbol: String,
    pub old: String,
    pub new: String,
    pub start_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub content: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResultCounts {
    pub files: usize,
    pub matches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMetadata {
    pub language: String,
    pub imports: usize,
    pub functions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDetail {
    None,
    Generic {
        input: Option<String>,
        result: Option<String>,
    },
    Shell {
        command: String,
        arguments: Vec<String>,
        workdir: Option<String>,
        output: Option<ShellOutput>,
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
    Edit {
        file: String,
        old: String,
        new: String,
        replace_all: bool,
        start_line: Option<usize>,
    },
    MultiEdit {
        file: String,
        edit_count: usize,
        sections: Vec<MultiEditSection>,
    },
    ReplaceSymbolInput {
        replacements: Vec<SymbolReplacement>,
    },
    #[allow(dead_code)]
    ReplaceSymbolDiff {
        replacements: Vec<SymbolReplacement>,
        sections: Vec<SymbolDiffSection>,
    },
    SearchText {
        pattern: String,
        path: String,
        include: String,
        counts: Option<SearchResultCounts>,
    },
    Glob {
        pattern: String,
        path: String,
    },
    List {
        path: String,
    },
    Index {
        path: String,
        metadata: Option<IndexMetadata>,
    },
    DeleteFile {
        path: String,
    },
    Browse {
        url: String,
    },
    Todo {
        items: Vec<TodoItem>,
    },
    Delegate {
        target_agent_id: String,
        objective: String,
    },
    LanguageQuery {
        action: String,
        uri: String,
    },
    Question {
        prompt: String,
    },
    ApplyPatch {
        patch: String,
    },
}
