.PHONY: build build-metal build-cuda push version patch minor major alpha beta

BUILD_DIR := ./target/release
CARGO_TOML := ./Cargo.toml

# Get current version from latest git tag (strips 'v' prefix)
CURRENT_VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.1.0")

# Build with Metal acceleration (Apple Silicon)
build-metal:
	cargo build --release --features metal

# Build with CUDA acceleration (NVIDIA GPU)
build-cuda:
	cargo build --release --features cuda

# Default build (auto-detect platform)
build:
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "Building with Metal support..."; \
		cargo build --release --features metal; \
	elif command -v nvcc >/dev/null 2>&1; then \
		echo "Building with CUDA support..."; \
		cargo build --release --features cuda; \
	else \
		echo "Error: No GPU acceleration available. Use 'make build-metal' or 'make build-cuda'."; \
		exit 1; \
	fi

push:
	git push origin main --tags

# Version bump: make version <patch|minor|major|alpha|beta>
version:
	@if [ -z "$(filter patch minor major alpha beta,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make version <patch|minor|major|alpha|beta>"; \
		echo "Current version: $(CURRENT_VERSION)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make version patch  # 0.1.0 -> 0.1.1"; \
		echo "  make version minor  # 0.1.0 -> 0.2.0"; \
		echo "  make version major  # 0.1.0 -> 1.0.0"; \
		echo "  make version alpha  # 0.1.0 -> 0.1.1-alpha.1 or 0.1.1-alpha.1 -> 0.1.1-alpha.2"; \
		echo "  make version beta   # 0.1.0 -> 0.1.1-beta.1 or 0.1.1-alpha.1 -> 0.1.1-beta.1"; \
		exit 1; \
	fi

patch minor major: version
	@TYPE=$@ && \
	echo "Current version: $(CURRENT_VERSION)" && \
	BASE_VERSION=$$(echo "$(CURRENT_VERSION)" | sed 's/-.*//') && \
	NEW_VERSION=$$(echo "$$BASE_VERSION" | awk -F. -v type="$$TYPE" '{ \
		if (type == "major") { print $$1+1".0.0" } \
		else if (type == "minor") { print $$1"."$$2+1".0" } \
		else { print $$1"."$$2"."$$3+1 } \
	}') && \
	echo "New version: $$NEW_VERSION" && \
	sed -i '' 's/^version = ".*"/version = "'$$NEW_VERSION'"/' $(CARGO_TOML) && \
	git add $(CARGO_TOML) && \
	git commit -m "chore: bump version to v$$NEW_VERSION" && \
	git tag "v$$NEW_VERSION" && \
	echo "Created tag v$$NEW_VERSION" && \
	echo "Run 'make push' to push changes and trigger release"

alpha beta: version
	@TYPE=$@ && \
	echo "Current version: $(CURRENT_VERSION)" && \
	if echo "$(CURRENT_VERSION)" | grep -q "\-$$TYPE\."; then \
		BASE=$$(echo "$(CURRENT_VERSION)" | sed "s/-$$TYPE\.[0-9]*//") && \
		NUM=$$(echo "$(CURRENT_VERSION)" | sed "s/.*-$$TYPE\.\([0-9]*\)/\1/") && \
		NEW_NUM=$$((NUM + 1)) && \
		NEW_VERSION="$$BASE-$$TYPE.$$NEW_NUM"; \
	elif echo "$(CURRENT_VERSION)" | grep -q "\-alpha\.\|\\-beta\."; then \
		BASE=$$(echo "$(CURRENT_VERSION)" | sed 's/-[a-z]*\.[0-9]*//') && \
		NEW_VERSION="$$BASE-$$TYPE.1"; \
	else \
		BASE_VERSION=$$(echo "$(CURRENT_VERSION)" | sed 's/-.*//') && \
		NEXT_PATCH=$$(echo "$$BASE_VERSION" | awk -F. '{ print $$1"."$$2"."$$3+1 }') && \
		NEW_VERSION="$$NEXT_PATCH-$$TYPE.1"; \
	fi && \
	echo "New version: $$NEW_VERSION" && \
	sed -i '' 's/^version = ".*"/version = "'$$NEW_VERSION'"/' $(CARGO_TOML) && \
	git add $(CARGO_TOML) && \
	git commit -m "chore: bump version to v$$NEW_VERSION" && \
	git tag "v$$NEW_VERSION" && \
	echo "Created tag v$$NEW_VERSION" && \
	echo "Run 'make push' to push changes and trigger release"
