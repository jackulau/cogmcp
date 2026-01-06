.PHONY: build release install test clean setup

VERSION := $(shell grep -m1 'version' Cargo.toml | cut -d'"' -f2)
BINARY := cogmcp
INSTALL_DIR := $(HOME)/.local/bin

# Build debug version
build:
	cargo build

# Build release version
release:
	cargo build --release

# Install to ~/.local/bin
install: release
	@mkdir -p $(INSTALL_DIR)
	@cp target/release/$(BINARY) $(INSTALL_DIR)/
	@echo "Installed $(BINARY) to $(INSTALL_DIR)"
	@echo "Run 'cogmcp setup' to configure"

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Run setup after install
setup:
	@$(INSTALL_DIR)/$(BINARY) setup

# Show help
help:
	@echo "CogMCP v$(VERSION)"
	@echo ""
	@echo "Usage:"
	@echo "  make build    - Build debug version"
	@echo "  make release  - Build release version"
	@echo "  make install  - Build and install to ~/.local/bin"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make setup    - Run setup after install"
