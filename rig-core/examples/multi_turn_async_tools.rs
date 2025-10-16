use futures::{Future, Stream, StreamExt};
use rig::{
    OneOrMany,
    agent::Agent,
    client::{CompletionClient, ProviderClient},
    completion::{self, CompletionError, CompletionModel, PromptError, ToolDefinition},
    message::{AssistantContent, Message, Text, ToolResultContent, UserContent},
    providers::anthropic,
    streaming::StreamingCompletion,
    tool::{Tool, ToolError, ToolSetError},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::pin::Pin;
use thiserror::Error;

#[derive(Debug, Error)]
enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] PromptError),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
}

#[derive(Debug, Error)]
#[error("Tool call receiver canceled")]
struct ReceiverCanceledError;

type StreamingResult = Pin<Box<dyn Stream<Item = Result<Text, StreamingError>> + Send>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let anthropic_client = anthropic::Client::from_env();

    let calculator_agent = anthropic_client
        .agent(anthropic::CLAUDE_3_7_SONNET)
        .preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely. 
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context. 
            3. This is very important: never perform the operation yourself. 
            ",
        )
        .max_tokens(1024)
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        .build();

    let mut stream = multi_turn_prompt(
        calculator_agent,
        "Calculate 2 * (3 + 5) / 9  = ?. Describe the result to me.",
        Vec::new(),
    )
    .await;

    custom_stream_to_stdout(&mut stream).await?;

    Ok(())
}

async fn multi_turn_prompt<M>(
    agent: Agent<M>,
    prompt: impl Into<Message> + Send,
    mut chat_history: Vec<completion::Message>,
) -> StreamingResult
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: std::marker::Send,
{
    let prompt: Message = prompt.into();

    (Box::pin(async_stream::stream! {
        let mut current_prompt = prompt;
        let mut pending_tool_receivers: Vec<(
            AssistantContent,
            String,
            Pin<Box<dyn Future<Output = Result<String, ToolSetError>> + Send>>,
        )> = Vec::new();

        'outer: loop {
            let mut stream = agent
                .stream_completion(current_prompt.clone(), chat_history.clone())
                .await?
                .stream()
                .await?;

            chat_history.push(current_prompt.clone());

            let mut tool_calls = vec![];
            let mut tool_results = vec![];
            let mut did_call_tool = false;

            while let Some(content) = stream.next().await {
                match content {
                    Ok(AssistantContent::Text(text)) => {
                        yield Ok(Text { text: text.text });
                    },
                    Ok(AssistantContent::ToolCall(tool_call)) => {
                        let receiver = agent
                            .tools
                            .call_async(
                                &tool_call.function.name,
                                tool_call.function.arguments.to_string(),
                            )
                            .await;

                        let call_id = tool_call.id.clone();
                        let tool_name = tool_call.function.name.clone();
                        let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                        let result_future = Box::pin(async move {
                            receiver
                                .await
                                .map_err(|_| {
                                    ToolSetError::from(ToolError::ToolCallError(Box::new(
                                        ReceiverCanceledError,
                                    )))
                                })?
                        });

                        pending_tool_receivers.push((tool_call_msg, call_id, result_future));

                        yield Ok(Text {
                            text: format!("Awaiting tool `{tool_name}` …"),
                        });

                        did_call_tool = true;
                    },
                    Err(e) => {
                        yield Err(e.into());
                        break 'outer;
                    }
                }
            }

            if !pending_tool_receivers.is_empty() {
                let mut drained = Vec::new();
                drained.append(&mut pending_tool_receivers);

                for (tool_call_msg, call_id, future) in drained {
                    tool_calls.push(tool_call_msg);
                    let tool_result = future.await?;
                    tool_results.push((call_id, tool_result));
                }
            }

            if !tool_calls.is_empty() {
                chat_history.push(Message::Assistant {
                    content: OneOrMany::many(tool_calls).expect("Impossible EmptyListError"),
                });
            }

            for (id, tool_result) in tool_results {
                chat_history.push(Message::User {
                    content: OneOrMany::one(UserContent::tool_result(
                        id,
                        OneOrMany::one(ToolResultContent::text(tool_result)),
                    )),
                });
            }

            current_prompt = match chat_history.pop() {
                Some(prompt) => prompt,
                None => unreachable!("Chat history should never be empty at this point"),
            };

            if !did_call_tool {
                break;
            }
        }

    })) as _
}

async fn custom_stream_to_stdout(stream: &mut StreamingResult) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(Text { text }) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
        }
    }
    println!();

    Ok(())
}

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Add;
impl Tool for Add {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "add",
            "description": "Add x and y together",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Multiply;
impl Tool for Multiply {
    const NAME: &'static str = "multiply";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "multiply",
            "description": "Multiply x and y",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to multiply"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to multiply"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x * args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Divide;
impl Tool for Divide {
    const NAME: &'static str = "divide";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "divide",
            "description": "Divide x by y",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to divide"
                    },
                    "y": {
                        "type": "number",
                        "description": "The divisor"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.y == 0 {
            Err(MathError)
        } else {
            Ok(args.x / args.y)
        }
    }
}
