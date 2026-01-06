//! MCP method constants and routing helpers
//!
//! This module defines the standard MCP method names and provides
//! utilities for method routing and categorization.

use std::fmt;

/// Current MCP protocol version
pub const MCP_VERSION: &str = "2024-11-05";

/// MCP method names as constants
pub mod method_names {
    // Lifecycle methods
    pub const INITIALIZE: &str = "initialize";
    pub const INITIALIZED: &str = "notifications/initialized";
    pub const PING: &str = "ping";
    pub const CANCELLED: &str = "notifications/cancelled";
    pub const PROGRESS: &str = "notifications/progress";

    // Tools methods
    pub const TOOLS_LIST: &str = "tools/list";
    pub const TOOLS_CALL: &str = "tools/call";

    // Resources methods
    pub const RESOURCES_LIST: &str = "resources/list";
    pub const RESOURCES_READ: &str = "resources/read";
    pub const RESOURCES_TEMPLATES_LIST: &str = "resources/templates/list";
    pub const RESOURCES_SUBSCRIBE: &str = "resources/subscribe";
    pub const RESOURCES_UNSUBSCRIBE: &str = "resources/unsubscribe";
    pub const RESOURCES_UPDATED: &str = "notifications/resources/updated";
    pub const RESOURCES_LIST_CHANGED: &str = "notifications/resources/list_changed";

    // Prompts methods
    pub const PROMPTS_LIST: &str = "prompts/list";
    pub const PROMPTS_GET: &str = "prompts/get";

    // Logging methods
    pub const LOGGING_SET_LEVEL: &str = "logging/setLevel";
    pub const LOGGING_MESSAGE: &str = "notifications/message";

    // Sampling methods (client-to-server)
    pub const SAMPLING_CREATE_MESSAGE: &str = "sampling/createMessage";

    // Roots methods (client-to-server)
    pub const ROOTS_LIST: &str = "roots/list";
    pub const ROOTS_LIST_CHANGED: &str = "notifications/roots/list_changed";

    // Completion methods
    pub const COMPLETION_COMPLETE: &str = "completion/complete";
}

/// Enumeration of all standard MCP methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum McpMethod {
    // Lifecycle
    Initialize,
    Initialized,
    Ping,
    Cancelled,
    Progress,

    // Tools
    ToolsList,
    ToolsCall,

    // Resources
    ResourcesList,
    ResourcesRead,
    ResourcesTemplatesList,
    ResourcesSubscribe,
    ResourcesUnsubscribe,
    ResourcesUpdated,
    ResourcesListChanged,

    // Prompts
    PromptsList,
    PromptsGet,

    // Logging
    LoggingSetLevel,
    LoggingMessage,

    // Sampling
    SamplingCreateMessage,

    // Roots
    RootsList,
    RootsListChanged,

    // Completion
    CompletionComplete,
}

impl McpMethod {
    /// Get the string representation of this method
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Initialize => method_names::INITIALIZE,
            Self::Initialized => method_names::INITIALIZED,
            Self::Ping => method_names::PING,
            Self::Cancelled => method_names::CANCELLED,
            Self::Progress => method_names::PROGRESS,
            Self::ToolsList => method_names::TOOLS_LIST,
            Self::ToolsCall => method_names::TOOLS_CALL,
            Self::ResourcesList => method_names::RESOURCES_LIST,
            Self::ResourcesRead => method_names::RESOURCES_READ,
            Self::ResourcesTemplatesList => method_names::RESOURCES_TEMPLATES_LIST,
            Self::ResourcesSubscribe => method_names::RESOURCES_SUBSCRIBE,
            Self::ResourcesUnsubscribe => method_names::RESOURCES_UNSUBSCRIBE,
            Self::ResourcesUpdated => method_names::RESOURCES_UPDATED,
            Self::ResourcesListChanged => method_names::RESOURCES_LIST_CHANGED,
            Self::PromptsList => method_names::PROMPTS_LIST,
            Self::PromptsGet => method_names::PROMPTS_GET,
            Self::LoggingSetLevel => method_names::LOGGING_SET_LEVEL,
            Self::LoggingMessage => method_names::LOGGING_MESSAGE,
            Self::SamplingCreateMessage => method_names::SAMPLING_CREATE_MESSAGE,
            Self::RootsList => method_names::ROOTS_LIST,
            Self::RootsListChanged => method_names::ROOTS_LIST_CHANGED,
            Self::CompletionComplete => method_names::COMPLETION_COMPLETE,
        }
    }

    /// Parse a method string into an McpMethod enum
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            method_names::INITIALIZE => Some(Self::Initialize),
            method_names::INITIALIZED => Some(Self::Initialized),
            method_names::PING => Some(Self::Ping),
            method_names::CANCELLED => Some(Self::Cancelled),
            method_names::PROGRESS => Some(Self::Progress),
            method_names::TOOLS_LIST => Some(Self::ToolsList),
            method_names::TOOLS_CALL => Some(Self::ToolsCall),
            method_names::RESOURCES_LIST => Some(Self::ResourcesList),
            method_names::RESOURCES_READ => Some(Self::ResourcesRead),
            method_names::RESOURCES_TEMPLATES_LIST => Some(Self::ResourcesTemplatesList),
            method_names::RESOURCES_SUBSCRIBE => Some(Self::ResourcesSubscribe),
            method_names::RESOURCES_UNSUBSCRIBE => Some(Self::ResourcesUnsubscribe),
            method_names::RESOURCES_UPDATED => Some(Self::ResourcesUpdated),
            method_names::RESOURCES_LIST_CHANGED => Some(Self::ResourcesListChanged),
            method_names::PROMPTS_LIST => Some(Self::PromptsList),
            method_names::PROMPTS_GET => Some(Self::PromptsGet),
            method_names::LOGGING_SET_LEVEL => Some(Self::LoggingSetLevel),
            method_names::LOGGING_MESSAGE => Some(Self::LoggingMessage),
            method_names::SAMPLING_CREATE_MESSAGE => Some(Self::SamplingCreateMessage),
            method_names::ROOTS_LIST => Some(Self::RootsList),
            method_names::ROOTS_LIST_CHANGED => Some(Self::RootsListChanged),
            method_names::COMPLETION_COMPLETE => Some(Self::CompletionComplete),
            _ => None,
        }
    }

    /// Check if this method is a notification (no response expected)
    pub fn is_notification(&self) -> bool {
        matches!(
            self,
            Self::Initialized
                | Self::Cancelled
                | Self::Progress
                | Self::ResourcesUpdated
                | Self::ResourcesListChanged
                | Self::LoggingMessage
                | Self::RootsListChanged
        )
    }

    /// Check if this method is a request (response expected)
    pub fn is_request(&self) -> bool {
        !self.is_notification()
    }

    /// Get the category of this method
    pub fn category(&self) -> MethodCategory {
        match self {
            Self::Initialize | Self::Initialized | Self::Ping | Self::Cancelled | Self::Progress => {
                MethodCategory::Lifecycle
            }
            Self::ToolsList | Self::ToolsCall => MethodCategory::Tools,
            Self::ResourcesList
            | Self::ResourcesRead
            | Self::ResourcesTemplatesList
            | Self::ResourcesSubscribe
            | Self::ResourcesUnsubscribe
            | Self::ResourcesUpdated
            | Self::ResourcesListChanged => MethodCategory::Resources,
            Self::PromptsList | Self::PromptsGet => MethodCategory::Prompts,
            Self::LoggingSetLevel | Self::LoggingMessage => MethodCategory::Logging,
            Self::SamplingCreateMessage => MethodCategory::Sampling,
            Self::RootsList | Self::RootsListChanged => MethodCategory::Roots,
            Self::CompletionComplete => MethodCategory::Completion,
        }
    }
}

impl fmt::Display for McpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Categories of MCP methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodCategory {
    Lifecycle,
    Tools,
    Resources,
    Prompts,
    Logging,
    Sampling,
    Roots,
    Completion,
}

impl MethodCategory {
    /// Get the string representation of this category
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lifecycle => "lifecycle",
            Self::Tools => "tools",
            Self::Resources => "resources",
            Self::Prompts => "prompts",
            Self::Logging => "logging",
            Self::Sampling => "sampling",
            Self::Roots => "roots",
            Self::Completion => "completion",
        }
    }
}

impl fmt::Display for MethodCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_as_str() {
        assert_eq!(McpMethod::Initialize.as_str(), "initialize");
        assert_eq!(McpMethod::ToolsList.as_str(), "tools/list");
        assert_eq!(McpMethod::ResourcesRead.as_str(), "resources/read");
    }

    #[test]
    fn test_method_from_str() {
        assert_eq!(McpMethod::from_str("initialize"), Some(McpMethod::Initialize));
        assert_eq!(McpMethod::from_str("tools/list"), Some(McpMethod::ToolsList));
        assert_eq!(McpMethod::from_str("unknown"), None);
    }

    #[test]
    fn test_method_is_notification() {
        assert!(McpMethod::Initialized.is_notification());
        assert!(McpMethod::Cancelled.is_notification());
        assert!(!McpMethod::Initialize.is_notification());
        assert!(!McpMethod::ToolsCall.is_notification());
    }

    #[test]
    fn test_method_category() {
        assert_eq!(McpMethod::Initialize.category(), MethodCategory::Lifecycle);
        assert_eq!(McpMethod::ToolsCall.category(), MethodCategory::Tools);
        assert_eq!(McpMethod::ResourcesRead.category(), MethodCategory::Resources);
    }

    #[test]
    fn test_all_methods_round_trip() {
        let methods = [
            McpMethod::Initialize,
            McpMethod::Initialized,
            McpMethod::Ping,
            McpMethod::ToolsList,
            McpMethod::ToolsCall,
            McpMethod::ResourcesList,
            McpMethod::ResourcesRead,
            McpMethod::PromptsList,
            McpMethod::PromptsGet,
        ];

        for method in methods {
            let s = method.as_str();
            let parsed = McpMethod::from_str(s);
            assert_eq!(parsed, Some(method), "Round-trip failed for {:?}", method);
        }
    }
}
