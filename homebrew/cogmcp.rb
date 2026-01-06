# Homebrew formula for CogMCP
# To use: brew install cogmcp/tap/cogmcp

class Contextmcp < Formula
  desc "Local-first MCP server for AI coding assistants"
  homepage "https://github.com/cogmcp/cogmcp"
  version "0.1.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/cogmcp/cogmcp/releases/download/v0.1.0/cogmcp-darwin-aarch64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_ARM64"
    end
    on_intel do
      url "https://github.com/cogmcp/cogmcp/releases/download/v0.1.0/cogmcp-darwin-x86_64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_X64"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/cogmcp/cogmcp/releases/download/v0.1.0/cogmcp-linux-aarch64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_LINUX_ARM64"
    end
    on_intel do
      url "https://github.com/cogmcp/cogmcp/releases/download/v0.1.0/cogmcp-linux-x86_64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_LINUX_X64"
    end
  end

  def install
    bin.install "cogmcp"
  end

  def caveats
    <<~EOS
      To configure CogMCP for your project, run:
        cd /path/to/your/project
        cogmcp setup

      Then add the generated configuration to your IDE's MCP settings.
    EOS
  end

  test do
    assert_match "cogmcp", shell_output("#{bin}/cogmcp --version")
  end
end
