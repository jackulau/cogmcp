//! Stdio transport implementation for MCP
//!
//! Provides async stdin/stdout transport for reading and writing
//! newline-delimited JSON messages.

use serde::{de::DeserializeOwned, Serialize};
use std::io;
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter, Stdin, Stdout,
};
use tracing::trace;

/// Error type for transport operations
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// IO error during read/write
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Connection closed (EOF reached)
    #[error("Connection closed")]
    ConnectionClosed,
}

/// Result type for transport operations
pub type TransportResult<T> = Result<T, TransportError>;

/// Async transport for reading/writing newline-delimited JSON over stdio
///
/// This transport handles the low-level I/O for MCP communication without
/// knowledge of the protocol semantics. Messages are read and written as
/// single lines of JSON text.
pub struct StdioTransport<R, W>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    reader: R,
    writer: BufWriter<W>,
}

impl StdioTransport<BufReader<Stdin>, Stdout> {
    /// Create a new transport using actual stdin/stdout
    pub fn new() -> Self {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        Self::from_handles(BufReader::new(stdin), stdout)
    }
}

impl Default for StdioTransport<BufReader<Stdin>, Stdout> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R, W> StdioTransport<R, W>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Create a transport from custom reader/writer handles
    ///
    /// This is useful for testing with mock I/O.
    pub fn from_handles(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer: BufWriter::new(writer),
        }
    }

    /// Read the next JSON message from stdin
    ///
    /// Returns `Ok(Some(value))` when a message is successfully read and parsed.
    /// Returns `Ok(None)` when EOF is reached (connection closed gracefully).
    /// Returns `Err` on I/O errors or malformed JSON.
    pub async fn read_message<T>(&mut self) -> TransportResult<Option<T>>
    where
        T: DeserializeOwned,
    {
        loop {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line).await?;

            if bytes_read == 0 {
                trace!("EOF reached on transport input");
                return Ok(None);
            }

            // Trim the trailing newline
            let trimmed = line.trim_end();

            if trimmed.is_empty() {
                // Empty line, skip and try to read next
                trace!("Skipping empty line");
                continue;
            }

            trace!(message = %trimmed, "Received message");

            let message: T = serde_json::from_str(trimmed)?;
            return Ok(Some(message));
        }
    }

    /// Read the next raw JSON string from stdin
    ///
    /// Returns the raw JSON string without parsing it into a specific type.
    /// This is useful when you need to inspect the message before deciding
    /// how to parse it.
    pub async fn read_raw_message(&mut self) -> TransportResult<Option<String>> {
        loop {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line).await?;

            if bytes_read == 0 {
                trace!("EOF reached on transport input");
                return Ok(None);
            }

            let trimmed = line.trim_end().to_string();

            if trimmed.is_empty() {
                trace!("Skipping empty line");
                continue;
            }

            trace!(message = %trimmed, "Received raw message");
            return Ok(Some(trimmed));
        }
    }

    /// Write a message to stdout as newline-delimited JSON
    ///
    /// The message is serialized to JSON and written followed by a newline.
    /// The output is flushed immediately to ensure prompt delivery.
    pub async fn write_message<T>(&mut self, message: &T) -> TransportResult<()>
    where
        T: Serialize,
    {
        let json = serde_json::to_string(message)?;
        trace!(message = %json, "Sending message");

        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        Ok(())
    }

    /// Write a raw JSON string to stdout
    ///
    /// The string is written as-is followed by a newline.
    /// Caller is responsible for ensuring the string is valid JSON.
    pub async fn write_raw_message(&mut self, json: &str) -> TransportResult<()> {
        trace!(message = %json, "Sending raw message");

        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        Ok(())
    }

    /// Flush any buffered output
    pub async fn flush(&mut self) -> TransportResult<()> {
        self.writer.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to create a transport with mock input/output
    fn mock_transport(input: &str) -> StdioTransport<BufReader<Cursor<Vec<u8>>>, Vec<u8>> {
        let reader = BufReader::new(Cursor::new(input.as_bytes().to_vec()));
        let writer = Vec::new();
        StdioTransport::from_handles(reader, writer)
    }

    /// Helper to get the written output from a transport
    fn get_output(transport: StdioTransport<BufReader<Cursor<Vec<u8>>>, Vec<u8>>) -> String {
        String::from_utf8(transport.writer.into_inner()).unwrap()
    }

    #[tokio::test]
    async fn test_read_message_success() {
        let input = r#"{"id":1,"method":"test"}"#.to_string() + "\n";
        let mut transport = mock_transport(&input);

        let message: Option<serde_json::Value> = transport.read_message().await.unwrap();
        assert!(message.is_some());

        let msg = message.unwrap();
        assert_eq!(msg["id"], 1);
        assert_eq!(msg["method"], "test");
    }

    #[tokio::test]
    async fn test_read_message_eof() {
        let mut transport = mock_transport("");

        let message: Option<serde_json::Value> = transport.read_message().await.unwrap();
        assert!(message.is_none());
    }

    #[tokio::test]
    async fn test_read_message_skips_empty_lines() {
        let input = "\n\n".to_string() + r#"{"id":1}"# + "\n";
        let mut transport = mock_transport(&input);

        let message: Option<serde_json::Value> = transport.read_message().await.unwrap();
        assert!(message.is_some());
        assert_eq!(message.unwrap()["id"], 1);
    }

    #[tokio::test]
    async fn test_read_message_malformed_json() {
        let input = "not valid json\n";
        let mut transport = mock_transport(input);

        let result: TransportResult<Option<serde_json::Value>> = transport.read_message().await;
        assert!(result.is_err());

        match result.unwrap_err() {
            TransportError::Json(_) => {}
            e => panic!("Expected JSON error, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_read_raw_message() {
        let input = r#"{"id":1,"method":"test"}"#.to_string() + "\n";
        let mut transport = mock_transport(&input);

        let message = transport.read_raw_message().await.unwrap();
        assert!(message.is_some());
        assert_eq!(message.unwrap(), r#"{"id":1,"method":"test"}"#);
    }

    #[tokio::test]
    async fn test_write_message() {
        let mut transport = mock_transport("");

        #[derive(Serialize)]
        struct TestMessage {
            id: i32,
            method: String,
        }

        let msg = TestMessage {
            id: 1,
            method: "test".to_string(),
        };

        transport.write_message(&msg).await.unwrap();
        let output = get_output(transport);

        assert_eq!(output, "{\"id\":1,\"method\":\"test\"}\n");
    }

    #[tokio::test]
    async fn test_write_raw_message() {
        let mut transport = mock_transport("");

        transport
            .write_raw_message(r#"{"custom":"json"}"#)
            .await
            .unwrap();
        let output = get_output(transport);

        assert_eq!(output, "{\"custom\":\"json\"}\n");
    }

    #[tokio::test]
    async fn test_multiple_messages() {
        let input = r#"{"id":1}"#.to_string() + "\n" + r#"{"id":2}"# + "\n" + r#"{"id":3}"# + "\n";
        let mut transport = mock_transport(&input);

        for expected_id in 1..=3 {
            let message: Option<serde_json::Value> = transport.read_message().await.unwrap();
            assert!(message.is_some());
            assert_eq!(message.unwrap()["id"], expected_id);
        }

        // Should hit EOF
        let message: Option<serde_json::Value> = transport.read_message().await.unwrap();
        assert!(message.is_none());
    }

    #[tokio::test]
    async fn test_read_and_write_roundtrip() {
        let original = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {}
            }
        });

        // Write message
        let mut write_transport = mock_transport("");
        write_transport.write_message(&original).await.unwrap();
        let json_line = get_output(write_transport);

        // Read it back
        let mut read_transport = mock_transport(&json_line);
        let read_back: Option<serde_json::Value> = read_transport.read_message().await.unwrap();

        assert!(read_back.is_some());
        assert_eq!(read_back.unwrap(), original);
    }
}
