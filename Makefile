.PHONY: build test test-e2e check recipe-schema push version patch minor major alpha beta

BUILD_DIR := ./target/release
CARGO_TOML := ./Cargo.toml

# Get current version from latest git tag (strips 'v' prefix)
CURRENT_VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.1.0")

# Release build of the runtime (`ohmygpu-runtime`) and the CLI (`ohmygpu`, alias `omg`).
# No GPU toolchain needed: llama.cpp binaries are fetched at runtime per platform.
build:
	cargo build --release
	@echo "Built $(BUILD_DIR)/ohmygpu-runtime and $(BUILD_DIR)/ohmygpu"

# Fast unit + API tests (mock backend; no GPU, no network)
test:
	cargo test --workspace

# Real llama.cpp end-to-end test (downloads llama.cpp + a ~470 MB model)
test-e2e:
	OHMYGPU_E2E=1 cargo test -p ohmygpu_daemon --test e2e_llamacpp -- --ignored --nocapture

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

# Regenerate schemas/recipe-v1.json from the Rust types in crates/core/src/recipe.rs
# (a unit test fails when the checked-in file is stale).
recipe-schema:
	cargo test -p ohmygpu_core recipe::tests::regenerate_schema -- --ignored

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
	cargo update -q --workspace && \
	git add $(CARGO_TOML) Cargo.lock && \
	git commit -m "chore: bump version to v$$NEW_VERSION" && \
	git tag "v$$NEW_VERSION" && \
	echo "Created tag v$$NEW_VERSION" && \
	echo "Run 'make push' to push the tag; .github/workflows/release.yml builds and publishes the binaries"

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
	cargo update -q --workspace && \
	git add $(CARGO_TOML) Cargo.lock && \
	git commit -m "chore: bump version to v$$NEW_VERSION" && \
	git tag "v$$NEW_VERSION" && \
	echo "Created tag v$$NEW_VERSION" && \
	echo "Run 'make push' to push the tag; .github/workflows/release.yml builds and publishes the binaries"
