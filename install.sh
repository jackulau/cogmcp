#!/bin/bash
# CogMCP Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/cogmcp/cogmcp/main/install.sh | bash

set -e

VERSION="${COGMCP_VERSION:-latest}"
INSTALL_DIR="${COGMCP_INSTALL_DIR:-$HOME/.local/bin}"
REPO="cogmcp/cogmcp"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux) OS="linux" ;;
        darwin) OS="darwin" ;;
        *) error "Unsupported OS: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="x86_64" ;;
        arm64|aarch64) ARCH="aarch64" ;;
        *) error "Unsupported architecture: $ARCH" ;;
    esac

    PLATFORM="${OS}-${ARCH}"
    info "Detected platform: $PLATFORM"
}

# Get latest version from GitHub
get_latest_version() {
    if [ "$VERSION" = "latest" ]; then
        VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v?([^"]+)".*/\1/')
        if [ -z "$VERSION" ]; then
            error "Could not determine latest version"
        fi
    fi
    info "Installing version: $VERSION"
}

# Build from source if no release available
build_from_source() {
    info "Building from source..."

    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        warn "Rust not found. Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Clone and build
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    info "Cloning repository..."
    git clone --depth 1 "https://github.com/$REPO.git" cogmcp
    cd cogmcp

    info "Building release binary..."
    cargo build --release

    # Install
    mkdir -p "$INSTALL_DIR"
    cp target/release/cogmcp "$INSTALL_DIR/"

    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"

    success "Built and installed cogmcp to $INSTALL_DIR/cogmcp"
}

# Download pre-built binary
download_binary() {
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/v$VERSION/cogmcp-$PLATFORM.tar.gz"

    info "Downloading from $DOWNLOAD_URL..."

    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    if curl -fsSL "$DOWNLOAD_URL" -o cogmcp.tar.gz 2>/dev/null; then
        tar -xzf cogmcp.tar.gz
        mkdir -p "$INSTALL_DIR"
        mv cogmcp "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/cogmcp"
        rm -rf "$TEMP_DIR"
        success "Downloaded and installed cogmcp to $INSTALL_DIR/cogmcp"
    else
        warn "No pre-built binary available for $PLATFORM"
        cd /
        rm -rf "$TEMP_DIR"
        build_from_source
    fi
}

# Add to PATH if needed
setup_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        warn "$INSTALL_DIR is not in your PATH"

        SHELL_NAME=$(basename "$SHELL")
        case "$SHELL_NAME" in
            bash) RC_FILE="$HOME/.bashrc" ;;
            zsh) RC_FILE="$HOME/.zshrc" ;;
            fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
            *) RC_FILE="" ;;
        esac

        if [ -n "$RC_FILE" ]; then
            echo "" >> "$RC_FILE"
            if [ "$SHELL_NAME" = "fish" ]; then
                echo "set -gx PATH $INSTALL_DIR \$PATH" >> "$RC_FILE"
            else
                echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$RC_FILE"
            fi
            success "Added $INSTALL_DIR to PATH in $RC_FILE"
            warn "Run 'source $RC_FILE' or restart your terminal"
        else
            warn "Add this to your shell config: export PATH=\"$INSTALL_DIR:\$PATH\""
        fi
    fi
}

# Main installation
main() {
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║     CogMCP Installer              ║"
    echo "║     Local-first MCP Context Server    ║"
    echo "╚═══════════════════════════════════════╝"
    echo ""

    detect_platform
    get_latest_version
    download_binary
    setup_path

    echo ""
    success "Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'cogmcp setup' to generate MCP configuration"
    echo "  2. Add the config to your Claude Code settings"
    echo "  3. Restart Claude Code"
    echo ""
    echo "Quick start:"
    echo "  cd /path/to/your/project"
    echo "  cogmcp setup"
    echo ""
}

main "$@"
