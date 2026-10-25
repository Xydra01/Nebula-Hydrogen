//! Model profiles: which runtime, weights, context and flags each llama-server launch uses.
//! Values come from ADR-004 and ADR-006 via the `[model]` section of `config/default.toml`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nebula_proto::ReasoningEffort;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::ModelError;

/// The inference backend used to communicate with the model server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelBackendKind {
    /// Standard llama-server HTTP backend (default).
    #[default]
    LlamaServer,
    /// Fast Win32 shared memory Hydrogen backend.
    Hydrogen,
}

impl std::fmt::Display for ModelBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LlamaServer => write!(f, "llama-server"),
            Self::Hydrogen => write!(f, "hydrogen"),
        }
    }
}

/// The `[model]` config section.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Profile loaded at startup.
    pub default_profile: String,
    /// The inference backend kind (`llama-server` or `hydrogen`). Defaults to `llama-server`.
    #[serde(default)]
    pub backend: ModelBackendKind,
    /// llama-server executable per runtime name (`llama-prism`, `llama-stock`).
    pub runtimes: BTreeMap<String, PathBuf>,
    /// Profiles by name.
    pub profiles: BTreeMap<String, ModelProfile>,
}

impl ModelConfig {
    /// Looks up a profile.
    ///
    /// # Errors
    /// [`ModelError::UnknownProfile`].
    pub fn profile(&self, name: &str) -> Result<&ModelProfile, ModelError> {
        self.profiles
            .get(name)
            .ok_or_else(|| ModelError::UnknownProfile(name.to_owned()))
    }

    /// The executable for a profile's runtime.
    ///
    /// # Errors
    /// [`ModelError::Config`] if the runtime is not listed.
    pub fn runtime_for(&self, profile: &ModelProfile) -> Result<&PathBuf, ModelError> {
        self.runtimes
            .get(&profile.runtime)
            .ok_or_else(|| ModelError::Config(format!("unknown runtime {:?}", profile.runtime)))
    }

    /// Checks that the default profile exists and every profile names a known runtime.
    ///
    /// # Errors
    /// [`ModelError::UnknownProfile`] or [`ModelError::Config`].
    pub fn validate(&self) -> Result<(), ModelError> {
        self.profile(&self.default_profile)?;
        for p in self.profiles.values() {
            self.runtime_for(p)?;
        }
        Ok(())
    }
}

/// How a model family switches reasoning on and off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningStyle {
    /// Bonsai: `reasoning_effort` = none/low/medium/xhigh.
    #[default]
    Effort,
    /// Qwen-style templates: `enable_thinking` true/false.
    EnableThinking,
}

/// Sampling presets merged into each request body.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    /// Used when reasoning is on.
    #[serde(default)]
    pub thinking: Map<String, Value>,
    /// Used when reasoning is off.
    #[serde(default)]
    pub instruct: Map<String, Value>,
}

/// One llama-server launch configuration.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    /// Key into [`ModelConfig::runtimes`].
    pub runtime: String,
    /// GGUF weights.
    pub model: PathBuf,
    /// Context length.
    pub ctx: u32,
    /// KV cache type for K and V (`f16`, `q8_0`, `q4_0`).
    pub kv_type: String,
    /// KV mean-centering bias file (`--kv-mean-center`); implies `LLAMA_ATTN_ROT_DISABLE=1`.
    #[serde(default)]
    pub kv_bias: Option<PathBuf>,
    /// Extra flags after the common ones.
    #[serde(default)]
    pub flags: Vec<String>,
    /// How reasoning is requested.
    #[serde(default)]
    pub reasoning_style: ReasoningStyle,
    /// Sampling presets.
    #[serde(default)]
    pub sampling: Sampling,
    /// Serves embeddings rather than chat.
    #[serde(default)]
    pub embedding: bool,
    /// Commit charge the server is expected to take, checked before loading. On WDDM,
    /// llama-server commits about as much system memory as it uses in VRAM.
    #[serde(default)]
    pub commit_estimate_mib: Option<u64>,
    /// Override inference backend kind for this profile, if any.
    #[serde(default)]
    pub backend: Option<ModelBackendKind>,
}

impl ModelProfile {
    /// Command-line arguments for a launch on `127.0.0.1:port`.
    #[must_use]
    pub fn args(&self, port: u16) -> Vec<String> {
        if self.backend == Some(ModelBackendKind::Hydrogen) {
            let mut args = vec![
                "-m".into(),
                self.model.display().to_string(),
                "-c".into(),
                self.ctx.to_string(),
            ];
            args.extend(self.flags.iter().cloned());
            return args;
        }

        let mut args = vec![
            "-m".into(),
            self.model.display().to_string(),
            "-c".into(),
            self.ctx.to_string(),
            "-ctk".into(),
            self.kv_type.clone(),
            "-ctv".into(),
            self.kv_type.clone(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
        ];
        args.extend(self.flags.iter().cloned());
        if let Some(bias) = &self.kv_bias {
            args.push("--kv-mean-center".into());
            args.push(bias.display().to_string());
        }
        args
    }

    /// Environment variables for a launch (not including the API key).
    #[must_use]
    pub fn env(&self) -> Vec<(String, String)> {
        // The KV bias is calibrated with K rotation off; inference must match.
        if self.kv_bias.is_some() {
            vec![("LLAMA_ATTN_ROT_DISABLE".into(), "1".into())]
        } else {
            Vec::new()
        }
    }

    /// Reasoning and sampling members for a chat request body, as the bench harness sends them.
    #[must_use]
    pub fn request_extras(&self, reasoning: ReasoningEffort) -> Map<String, Value> {
        let mut body = Map::new();
        let effort = match reasoning {
            ReasoningEffort::None => "none",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::Xhigh => "xhigh",
        };
        match self.reasoning_style {
            ReasoningStyle::Effort => {
                body.insert("reasoning_effort".into(), effort.into());
                body.insert(
                    "chat_template_kwargs".into(),
                    serde_json::json!({ "reasoning_effort": effort }),
                );
            }
            ReasoningStyle::EnableThinking => {
                body.insert(
                    "chat_template_kwargs".into(),
                    serde_json::json!({ "enable_thinking": reasoning != ReasoningEffort::None }),
                );
            }
        }
        let preset = if reasoning == ReasoningEffort::None {
            &self.sampling.instruct
        } else {
            &self.sampling.thinking
        };
        body.extend(preset.clone());
        body
    }
}
