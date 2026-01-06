# Homebrew formula for CogMCP
# To use: brew tap cogmcp/tap && brew install cogmcp

class Cogmcp < Formula
  desc "Local-first MCP server for AI coding assistants"
  homepage "https://github.com/cogmcp/cogmcp"
  url "https://github.com/cogmcp/cogmcp.git", tag: "v0.1.0"
  license "MIT"
  head "https://github.com/cogmcp/cogmcp.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "crates/cogmcp-server")
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
    assert_match "cogmcp #{version}", shell_output("#{bin}/cogmcp --version")
  end
end
