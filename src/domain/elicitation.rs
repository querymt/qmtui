use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ElicitationOption {
    pub value: serde_json::Value,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElicitationFieldKind {
    SingleSelect { options: Vec<ElicitationOption> },
    MultiSelect { options: Vec<ElicitationOption> },
    TextInput,
    NumberInput { integer: bool },
    BooleanToggle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElicitationField {
    pub name: String,
    pub title: String,
    pub description: Option<String>,
    pub required: bool,
    pub kind: ElicitationFieldKind,
}

#[derive(Debug, Clone)]
pub struct ElicitationState {
    pub elicitation_id: String,
    pub message: String,
    pub source: String,
    pub fields: Vec<ElicitationField>,
    /// Accumulated schema values (field name -> value).
    pub selected: HashMap<String, serde_json::Value>,
    /// Entered value for the active text or number field.
    pub text_input: String,
    /// Entered value for a custom select response.
    pub custom_input: String,
    /// Whether select fields may offer a free-form custom response.
    pub allow_custom: bool,
}

impl ElicitationState {
    /// Parse a JSON Schema `properties` object into a flat list of fields.
    /// Mirrors `parseSchema` in `ElicitationCard.tsx`.
    pub fn parse_schema(schema: &serde_json::Value) -> Vec<ElicitationField> {
        let Some(props) = schema
            .get("properties")
            .and_then(|properties| properties.as_object())
        else {
            return Vec::new();
        };
        let required: std::collections::HashSet<&str> = schema
            .get("required")
            .and_then(|required| required.as_array())
            .map(|required| required.iter().filter_map(|value| value.as_str()).collect())
            .unwrap_or_default();

        let mut fields = Vec::new();
        for (name, property) in props {
            let title = property
                .get("title")
                .and_then(|value| value.as_str())
                .unwrap_or(name)
                .to_string();
            let description = property
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            let typ = property
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or("string");

            let kind = if let Some(one_of) =
                property.get("oneOf").and_then(|value| value.as_array())
            {
                ElicitationFieldKind::SingleSelect {
                    options: one_of.iter().map(option_from_schema).collect(),
                }
            } else if let Some(values) = property.get("enum").and_then(|value| value.as_array()) {
                ElicitationFieldKind::SingleSelect {
                    options: values
                        .iter()
                        .map(|value| ElicitationOption {
                            value: value.clone(),
                            label: value.as_str().unwrap_or("").to_string(),
                            description: None,
                        })
                        .collect(),
                }
            } else if typ == "array" {
                let options = property
                    .get("items")
                    .and_then(|items| items.get("anyOf").or_else(|| items.get("oneOf")))
                    .and_then(|values| values.as_array());
                options.map_or(ElicitationFieldKind::TextInput, |options| {
                    ElicitationFieldKind::MultiSelect {
                        options: options.iter().map(option_from_schema).collect(),
                    }
                })
            } else if typ == "boolean" {
                ElicitationFieldKind::BooleanToggle
            } else if typ == "integer" {
                ElicitationFieldKind::NumberInput { integer: true }
            } else if typ == "number" {
                ElicitationFieldKind::NumberInput { integer: false }
            } else {
                ElicitationFieldKind::TextInput
            };

            fields.push(ElicitationField {
                name: name.clone(),
                title,
                description,
                required: required.contains(name.as_str()),
                kind,
            });
        }
        fields
    }

    /// Record a schema-provided single-select option.
    pub fn select_option(&mut self, field_index: usize, option_index: usize) {
        let Some(field) = self.fields.get(field_index) else {
            return;
        };
        if let ElicitationFieldKind::SingleSelect { options } = &field.kind
            && let Some(option) = options.get(option_index)
        {
            self.custom_input.clear();
            self.selected
                .insert(field.name.clone(), option.value.clone());
        }
    }

    /// Clear the selected schema value for a field before entering a custom response.
    pub fn clear_selection(&mut self, field_index: usize) {
        if let Some(field) = self.fields.get(field_index) {
            self.selected.remove(&field.name);
        }
    }

    /// Toggle a schema-provided multi-select option or a boolean field.
    pub fn toggle_option(&mut self, field_index: usize, option_index: usize) {
        let Some(field) = self.fields.get(field_index) else {
            return;
        };
        match &field.kind {
            ElicitationFieldKind::MultiSelect { options } => {
                if let Some(option) = options.get(option_index) {
                    self.custom_input.clear();
                    let values = self
                        .selected
                        .entry(field.name.clone())
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                    if let serde_json::Value::Array(values) = values {
                        if let Some(position) =
                            values.iter().position(|value| value == &option.value)
                        {
                            values.remove(position);
                        } else {
                            values.push(option.value.clone());
                        }
                    }
                }
            }
            ElicitationFieldKind::BooleanToggle => {
                let next = self
                    .selected
                    .get(&field.name)
                    .and_then(serde_json::Value::as_bool)
                    .map(|value| !value)
                    .unwrap_or(true);
                self.selected
                    .insert(field.name.clone(), serde_json::Value::Bool(next));
            }
            ElicitationFieldKind::SingleSelect { .. }
            | ElicitationFieldKind::TextInput
            | ElicitationFieldKind::NumberInput { .. } => {}
        }
    }

    /// Build the `content` object to send with an accept response.
    pub fn build_accept_content(&self, custom_response: Option<&str>) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for field in &self.fields {
            match &field.kind {
                ElicitationFieldKind::SingleSelect { .. }
                | ElicitationFieldKind::MultiSelect { .. } => {
                    if let Some(custom_response) = custom_response {
                        let custom = serde_json::Value::String(custom_response.trim().to_string());
                        let value =
                            if matches!(&field.kind, ElicitationFieldKind::MultiSelect { .. }) {
                                serde_json::Value::Array(vec![custom])
                            } else {
                                custom
                            };
                        object.insert(field.name.clone(), value);
                    } else if let Some(value) = self.selected.get(&field.name) {
                        object.insert(field.name.clone(), value.clone());
                    }
                }
                ElicitationFieldKind::TextInput => {
                    if !self.text_input.is_empty() {
                        object.insert(
                            field.name.clone(),
                            serde_json::Value::String(self.text_input.clone()),
                        );
                    }
                }
                ElicitationFieldKind::NumberInput { integer } => {
                    if !self.text_input.is_empty() {
                        let value = if *integer {
                            self.text_input
                                .parse::<i64>()
                                .map(|number| serde_json::json!(number))
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            self.text_input
                                .parse::<f64>()
                                .map(|number| serde_json::json!(number))
                                .unwrap_or(serde_json::Value::Null)
                        };
                        object.insert(field.name.clone(), value);
                    }
                }
                ElicitationFieldKind::BooleanToggle => {
                    if let Some(value) = self.selected.get(&field.name) {
                        object.insert(field.name.clone(), value.clone());
                    }
                }
            }
        }
        serde_json::Value::Object(object)
    }

    /// Returns true if all required fields have a value.
    pub fn is_valid(&self, custom_response: Option<&str>) -> bool {
        if let Some(custom_response) = custom_response {
            return !custom_response.trim().is_empty();
        }
        self.fields.iter().all(|field| match &field.kind {
            ElicitationFieldKind::SingleSelect { .. }
            | ElicitationFieldKind::MultiSelect { .. }
            | ElicitationFieldKind::BooleanToggle => {
                !field.required || self.selected.contains_key(&field.name)
            }
            ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                !field.required || !self.text_input.is_empty()
            }
        })
    }

    /// Constructor used by unit tests.
    #[cfg(test)]
    pub fn new_for_test(fields: Vec<ElicitationField>) -> Self {
        Self {
            elicitation_id: "test-id".into(),
            message: "Test question".into(),
            source: "builtin:question".into(),
            fields,
            selected: HashMap::new(),
            text_input: String::new(),
            custom_input: String::new(),
            allow_custom: true,
        }
    }
}

fn option_from_schema(option: &serde_json::Value) -> ElicitationOption {
    ElicitationOption {
        value: option
            .get("const")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        label: option
            .get("title")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        description: option
            .get("description")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, required: bool, kind: ElicitationFieldKind) -> ElicitationField {
        ElicitationField {
            name: name.into(),
            title: name.into(),
            description: None,
            required,
            kind,
        }
    }

    fn option(value: &str, label: &str) -> ElicitationOption {
        ElicitationOption {
            value: serde_json::json!(value),
            label: label.into(),
            description: None,
        }
    }

    #[test]
    fn parses_supported_schema_fields() {
        let fields = ElicitationState::parse_schema(&serde_json::json!({
            "properties": {
                "choice": { "oneOf": [{ "const": "a", "title": "Option A" }] },
                "tags": { "type": "array", "items": { "anyOf": [{ "const": "x", "title": "X" }] } },
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "confirm": { "type": "boolean" }
            },
            "required": ["choice", "name"]
        }));

        assert!(matches!(
            fields[0].kind,
            ElicitationFieldKind::SingleSelect { .. }
        ));
        assert!(matches!(
            fields[1].kind,
            ElicitationFieldKind::MultiSelect { .. }
        ));
        assert!(matches!(fields[2].kind, ElicitationFieldKind::TextInput));
        assert!(matches!(
            fields[3].kind,
            ElicitationFieldKind::NumberInput { integer: true }
        ));
        assert!(matches!(
            fields[4].kind,
            ElicitationFieldKind::BooleanToggle
        ));
        assert!(fields[0].required);
        assert!(fields[2].required);
    }

    #[test]
    fn selected_values_preserve_accept_content_and_validation() {
        let mut state = ElicitationState::new_for_test(vec![field(
            "tags",
            true,
            ElicitationFieldKind::MultiSelect {
                options: vec![option("x", "X"), option("z", "Z")],
            },
        )]);
        assert!(!state.is_valid(None));
        state.toggle_option(0, 0);
        state.toggle_option(0, 1);

        assert!(state.is_valid(None));
        assert_eq!(
            state.build_accept_content(None),
            serde_json::json!({ "tags": ["x", "z"] })
        );
    }

    #[test]
    fn custom_response_replaces_selection_and_is_trimmed() {
        let mut state = ElicitationState::new_for_test(vec![field(
            "choice",
            true,
            ElicitationFieldKind::SingleSelect {
                options: vec![option("a", "A")],
            },
        )]);
        state.select_option(0, 0);
        state.clear_selection(0);

        assert!(state.is_valid(Some("  another answer  ")));
        assert_eq!(
            state.build_accept_content(Some("  another answer  ")),
            serde_json::json!({ "choice": "another answer" })
        );
    }
}
