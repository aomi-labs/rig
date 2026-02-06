//! Anthropic API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::anthropic;
//!
//! let client = anthropic::Client::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_3_5_SONNET);
//! ```
//!
//! # Prompt Caching
//!
//! Anthropic's prompt caching can reduce costs by up to 90% for multi-turn conversations.
//! Enable automatic caching with [`with_prompt_caching()`](completion::CompletionModel::with_prompt_caching):
//!
//! ```ignore
//! let model = client.completion_model(anthropic::CLAUDE_3_5_SONNET)
//!     .with_prompt_caching();
//! ```
//!
//! For finer control, use the builder methods on content types:
//!
//! ```ignore
//! use rig::providers::anthropic::completion::{Content, SystemContent, CacheControl};
//!
//! // Cache specific content blocks
//! let cached_text = Content::text("Important context...").cached();
//! let cached_system = SystemContent::text("You are a helpful assistant.").cached();
//! ```
//!
//! See <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching> for more details.

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder};
pub use completion::{
    CacheControl, Content, SystemContent, ToolDefinition,
    CLAUDE_3_5_HAIKU, CLAUDE_3_5_SONNET, CLAUDE_3_7_SONNET, CLAUDE_4_OPUS, CLAUDE_4_SONNET,
};
