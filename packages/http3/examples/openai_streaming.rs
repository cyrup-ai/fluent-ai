//! OpenAI streaming example demonstrating TRUE streaming over HTTP3 arrays without futures
//! This example shows how to stream OpenAI chat completions using pure AsyncStream patterns

use std::io::{self, Write};

use fluent_ai_http3::{ContentType, Http3, HttpChunk, HttpStreamExt};
use serde::{Deserialize, Serialize};

// OpenAI API request types
#[derive(Serialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Serialize, Debug)]
struct Message {
    role: String,
    content: String,
}

// OpenAI API response types for streaming
#[derive(Deserialize, Debug, Default)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug, Default)]
struct Choice {
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct Delta {
    role: Option<String>,
    content: Option<String>,
}

// Error response type
#[derive(Deserialize, Debug, Default)]
struct OpenAIError {
    error: ErrorDetails,
}

#[derive(Deserialize, Debug, Default)]
struct ErrorDetails {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("⚠️  OPENAI_API_KEY not found in environment");
        println!("🧪 Using mock streaming example instead...");
        "mock-key".to_string()
    });

    let use_real_api = api_key != "mock-key";

    if use_real_api {
        println!("🚀 Starting OpenAI streaming chat completion...");
        stream_openai_chat(&api_key)?;
    } else {
        println!("🧪 Running mock streaming example...");
        mock_streaming_example()?;
    }

    Ok(())
}

/// Stream real OpenAI chat completion using HTTP3 pure streaming
fn stream_openai_chat(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Write a haiku about streaming data in Rust. Make it technical and beautiful."
                .to_string(),
        }],
        stream: true,
        max_tokens: Some(150),
        temperature: Some(0.7),
    };

    println!("📝 Request: {:#?}", request);
    println!("🌊 Starting stream...\n");

    // Create HTTP3 stream - NO FUTURES, pure AsyncStream
    let stream = Http3::json()
        .bearer_auth(api_key)
        .accept(ContentType::ApplicationJson)
        .body(&request)
        .post("https://api.openai.com/v1/chat/completions");

    // TRUE STREAMING: Process each chunk as it arrives over the network
    // This uses AsyncStream internally - no futures, no .await calls
    print!("🤖 Assistant: ");
    io::stdout().flush().unwrap();

    stream
        .on_chunk(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Extract Server-Sent Events data from chunk
                    if let HttpChunk::Body(ref body_bytes) = chunk {
                        let body_str = String::from_utf8_lossy(body_bytes);

                        // Parse SSE format: "data: {json}\n\n"
                        for line in body_str.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..]; // Remove "data: " prefix

                                if json_str == "[DONE]" {
                                    println!("\n\n✅ Stream completed!");
                                    break;
                                }

                                // Parse JSON chunk
                                match serde_json::from_str::<ChatCompletionChunk>(json_str) {
                                    Ok(completion_chunk) => {
                                        // Extract content from first choice
                                        if let Some(choice) = completion_chunk.choices.first() {
                                            if let Some(content) = &choice.delta.content {
                                                print!("{}", content);
                                                io::stdout().flush().unwrap();
                                            }

                                            // Check for completion
                                            if choice.finish_reason.is_some() {
                                                println!(
                                                    "\n\n🎯 Completion reason: {:?}",
                                                    choice.finish_reason
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        // Skip parsing errors for SSE metadata
                                        if !json_str.is_empty() && json_str != "\n" {
                                            println!("\n⚠️  JSON parse error: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    chunk
                }
                Err(e) => {
                    println!("\n❌ Stream error: {}", e);
                    // Convert error to BadChunk
                    fluent_ai_http3::BadChunk::from_err(e).into()
                }
            }
        })
        .collect_one_or_else(|e| {
            println!("❌ Collection error: {}", e);

            // Try to parse as OpenAI error response
            if let Ok(openai_error) = serde_json::from_str::<OpenAIError>(&e.to_string()) {
                println!("🚨 OpenAI Error: {}", openai_error.error.message);
                println!("   Type: {}", openai_error.error.error_type);
                if let Some(code) = openai_error.error.code {
                    println!("   Code: {}", code);
                }
            }

            ChatCompletionChunk::default()
        });

    Ok(())
}

/// Mock streaming example to demonstrate the streaming pattern
fn mock_streaming_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Creating mock streaming server...");

    // Start local mock server that sends streaming SSE data
    let mock_server = std::thread::spawn(|| {
        use std::io::prelude::*;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
        println!("🌍 Mock server listening on 127.0.0.1:8080");

        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    // Read the request (we don't parse it, just consume it)
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer);

                    // Send SSE streaming response
                    let response = "HTTP/1.1 200 OK\r\n\
                                  Content-Type: text/plain\r\n\
                                  Cache-Control: no-cache\r\n\
                                  Connection: keep-alive\r\n\r\n";

                    let _ = stream.write_all(response.as_bytes());

                    // Stream mock data chunks
                    let chunks = vec![
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Async\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" streams\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" flow\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" like\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" rivers,\\n\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Zero\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" allocation\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" beauty,\\n\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Rust's\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" memory\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\" safe.\"}}]}\n\n",
                        "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n\n",
                    ];

                    for chunk in chunks {
                        let _ = stream.write_all(chunk.as_bytes());
                        let _ = stream.flush();
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }

                    break; // Handle one request then exit
                }
                Err(_) => {}
            }
        }
    });

    // Give server time to start
    std::thread::sleep(std::time::Duration::from_millis(100));

    println!("🌊 Connecting to mock streaming endpoint...\n");

    // Create streaming request to mock server
    let mock_request = ChatCompletionRequest {
        model: "mock-gpt-4".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Generate a haiku".to_string(),
        }],
        stream: true,
        max_tokens: Some(50),
        temperature: Some(0.7),
    };

    print!("🤖 Mock Assistant: ");
    io::stdout().flush().unwrap();

    // Stream from mock server using pure AsyncStream pattern
    let stream = Http3::json()
        .accept(ContentType::ApplicationJson)
        .body(&mock_request)
        .post("http://127.0.0.1:8080/v1/chat/completions");

    // TRUE STREAMING: Each chunk processed as it arrives
    stream
        .on_chunk(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    if let HttpChunk::Body(ref body_bytes) = chunk {
                        let body_str = String::from_utf8_lossy(body_bytes);

                        // Parse SSE data lines
                        for line in body_str.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..];

                                if json_str == "[DONE]" {
                                    println!("\n\n✅ Mock stream completed!");
                                    break;
                                }

                                // Parse and display content
                                if let Ok(completion_chunk) =
                                    serde_json::from_str::<ChatCompletionChunk>(json_str)
                                {
                                    if let Some(choice) = completion_chunk.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            print!("{}", content);
                                            io::stdout().flush().unwrap();
                                        }

                                        if choice.finish_reason.is_some() {
                                            println!("\n🎯 Mock completion finished!");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    chunk
                }
                Err(e) => {
                    println!("\n❌ Mock stream error: {}", e);
                    fluent_ai_http3::BadChunk::from_err(e).into()
                }
            }
        })
        .collect_one_or_else(|e| {
            println!("❌ Mock collection error: {}", e);
            ChatCompletionChunk::default()
        });

    // Wait for mock server thread to finish
    let _ = mock_server.join();

    println!("\n🎭 Mock streaming example completed!");
    Ok(())
}
