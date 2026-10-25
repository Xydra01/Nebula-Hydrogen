#![allow(clippy::unwrap_used, clippy::panic, missing_docs)]

use std::path::Path;

use nebula_model::sse::ChunkParser;
use nebula_model::{ModelConfig, ModelError, StreamItem, ToolCall};
use nebula_proto::{ReasoningEffort, StopReason};
use serde_json::json;

pub fn default_config() -> ModelConfig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/default.toml");
    let mut table: toml::Table = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    table.remove("model").unwrap().try_into().unwrap()
}

#[test]
fn default_config_parses_and_standard_matches_the_benchmarked_launch() {
    let cfg = default_config();
    cfg.validate().unwrap();
    assert_eq!(cfg.default_profile, "standard");
    let standard = cfg.profile("standard").unwrap();
    // bench/profiles/pq2mtp.toml through bench/nebula_bench/server.py, minus the runtime path.
    let expected = [
        "-m",
        r"F:\Nebula\models\bonsai2-27b-mtp\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
        "-c",
        "32768",
        "-ctk",
        "q4_0",
        "-ctv",
        "q4_0",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "-ngl",
        "99",
        "-fa",
        "on",
        "-np",
        "1",
        "--jinja",
        "--reasoning",
        "auto",
        "--ctx-checkpoints",
        "32",
        "--cache-ram",
        "4096",
        "--cache-idle-slots",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--cors-origins",
        "http://nebula.invalid",
        "--no-cors-credentials",
        "-lv",
        "4",
    ];
    assert_eq!(standard.args(8080), expected);
    assert!(standard.env().is_empty());

    let long = cfg.profile("long").unwrap();
    assert_eq!(long.ctx, 131_072);
    assert_eq!(
        long.env(),
        [("LLAMA_ATTN_ROT_DISABLE".to_owned(), "1".to_owned())]
    );
    let args = long.args(1);
    assert_eq!(args[args.len() - 2], "--kv-mean-center");

    assert!(cfg.profile("embedding").unwrap().embedding);
    assert!(matches!(
        cfg.profile("quality"),
        Err(ModelError::UnknownProfile(_))
    ));

    let mut hydrogen_prof = standard.clone();
    hydrogen_prof.backend = Some(nebula_model::config::ModelBackendKind::Hydrogen);
    hydrogen_prof.flags = vec!["--shm-name".to_string(), "Local\\CustomShm".to_string()];
    let h_args = hydrogen_prof.args(8080);
    assert_eq!(
        h_args,
        vec![
            "-m".to_string(),
            hydrogen_prof.model.display().to_string(),
            "-c".to_string(),
            "32768".to_string(),
            "--shm-name".to_string(),
            "Local\\CustomShm".to_string(),
        ]
    );
}

#[test]
fn request_extras_follow_the_bench_client() {
    let cfg = default_config();
    let p = cfg.profile("standard").unwrap();
    let none = p.request_extras(ReasoningEffort::None);
    assert_eq!(none["reasoning_effort"], "none");
    assert_eq!(none["presence_penalty"], 1.5);
    let xhigh = p.request_extras(ReasoningEffort::Xhigh);
    assert_eq!(
        xhigh["chat_template_kwargs"],
        json!({ "reasoning_effort": "xhigh" })
    );
    assert_eq!(xhigh["temperature"], 1.0);
}

#[test]
fn config_rejects_unknown_runtime_and_fields() {
    let mut cfg = default_config();
    cfg.profiles.get_mut("lean").unwrap().runtime = "llama-nope".into();
    assert!(matches!(cfg.validate(), Err(ModelError::Config(_))));
    let bad: Result<ModelConfig, _> =
        toml::from_str("default_profile = 'a'\nruntimes = {}\nprofiles = {}\nsurprise = 1\n");
    assert!(bad.is_err());
}

#[test]
fn parser_yields_tokens_usage_then_done() {
    let mut p = ChunkParser::new();
    let mut items = Vec::new();
    for data in [
        json!({ "choices": [{ "delta": { "role": "assistant", "content": null } }] }).to_string(),
        json!({ "choices": [{ "delta": { "reasoning_content": "hmm" } }] }).to_string(),
        json!({ "choices": [{ "delta": { "content": "Hi" } }] }).to_string(),
        json!({ "choices": [{ "delta": {}, "finish_reason": "length" }],
                "timings": { "prompt_n": 5, "prompt_ms": 1.5, "predicted_n": 7, "predicted_ms": 2.0 } })
        .to_string(),
        "[DONE]".to_owned(),
    ] {
        items.extend(p.feed(&data).unwrap());
    }
    assert!(p.is_done());
    assert_eq!(
        items[..2],
        [
            StreamItem::Token {
                text: "hmm".into(),
                reasoning: true
            },
            StreamItem::Token {
                text: "Hi".into(),
                reasoning: false
            },
        ]
    );
    assert!(matches!(items[2], StreamItem::Usage(u) if u.prompt_n == 5 && u.predicted_n == 7));
    assert_eq!(items[3], StreamItem::Done(StopReason::Length));
    assert_eq!(items.len(), 4);
    assert!(p.finish().is_empty());
}

#[test]
fn parser_assembles_parallel_tool_calls_in_index_order() {
    let mut p = ChunkParser::new();
    let chunks = [
        json!({ "choices": [{ "delta": { "tool_calls": [
            { "index": 1, "id": "b", "function": { "name": "two", "arguments": "{}" } },
            { "index": 0, "id": "a", "function": { "name": "one", "arguments": "{\"x\":" } },
        ] } }] }),
        json!({ "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "1}" } },
        ] } }] }),
        json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
    ];
    for c in chunks {
        assert!(p.feed(&c.to_string()).unwrap().is_empty());
    }
    let end = p.finish();
    assert_eq!(
        end,
        [
            StreamItem::ToolCall(ToolCall {
                id: "a".into(),
                name: "one".into(),
                arguments: r#"{"x":1}"#.into()
            }),
            StreamItem::ToolCall(ToolCall {
                id: "b".into(),
                name: "two".into(),
                arguments: "{}".into()
            }),
            StreamItem::Done(StopReason::ToolCalls),
        ]
    );
}

#[test]
fn parser_reports_errors() {
    let mut p = ChunkParser::new();
    assert!(p.feed("not json").is_err());
    assert!(
        p.feed(r#"{"error":{"message":"context too long"}}"#)
            .is_err()
    );
    // [DONE] without a finish reason yields nothing; the caller treats that as an error.
    assert!(p.feed("[DONE]").unwrap().is_empty());
    assert!(!p.is_done());
}
