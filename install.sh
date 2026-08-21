#!/bin/sh
# OhMyGPU Runtime installer — https://github.com/ohmygpu/ohmygpu
#
#   curl -fsSL https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/install.sh | sh
#
# Installs the latest GitHub release of `ohmygpu-runtime` (the runtime) and
# `ohmygpu` (the CLI, plus an `omg` symlink) on macOS and Linux, verifies the
# SHA-256 checksum shipped with the release, and upgrades an existing install
# in place. Once installed, `omg upgrade` does the same from the CLI.
#
# Options (flag or environment variable):
#   --version <tag>   OHMYGPU_VERSION      release to install, e.g. v0.5.0 (default: latest)
#   --dir <path>      OHMYGPU_INSTALL_DIR  install directory (default: see below)
#   --no-sudo         OHMYGPU_NO_SUDO=1    never escalate with sudo
#   --force                                reinstall even if that version is already installed
#   -h, --help
#
#   curl -fsSL https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/install.sh | sh -s -- --version v0.5.0
#
# Install directory: --dir / OHMYGPU_INSTALL_DIR; else the directory of the
# `ohmygpu` already on PATH (upgrade in place); else /usr/local/bin (with sudo
# when needed); else ~/.local/bin.
#
# Windows: download ohmygpu-x86_64-pc-windows-msvc.zip from the releases page
# and put ohmygpu-runtime.exe and ohmygpu.exe on your PATH.

set -eu

REPO="ohmygpu/ohmygpu"
RELEASES="https://github.com/${REPO}/releases"
VERSION="${OHMYGPU_VERSION:-latest}"
INSTALL_DIR="${OHMYGPU_INSTALL_DIR:-}"
NO_SUDO="${OHMYGPU_NO_SUDO:-}"
FORCE=""

usage() {
    cat <<'EOF'
OhMyGPU Runtime installer

  curl -fsSL https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/install.sh | sh
  curl -fsSL https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/install.sh | sh -s -- [options]

Options (flag or environment variable):
  --version <tag>   OHMYGPU_VERSION      release to install, e.g. v0.5.0 (default: latest)
  --dir <path>      OHMYGPU_INSTALL_DIR  install directory (default: existing install dir,
                                         else /usr/local/bin, else ~/.local/bin)
  --no-sudo         OHMYGPU_NO_SUDO=1    never escalate with sudo
  --force                                reinstall even if that version is already installed
  -h, --help        this help

Installs ohmygpu-runtime, ohmygpu and the omg symlink from a GitHub release
(https://github.com/ohmygpu/ohmygpu/releases), verifying SHA256SUMS.txt.
EOF
}

info() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------

detect_target() {
    os=$(uname -s 2>/dev/null || echo unknown)
    arch=$(uname -m 2>/dev/null || echo unknown)
    case "$arch" in
        x86_64 | amd64) arch=x86_64 ;;
        arm64 | aarch64) arch=aarch64 ;;
        *) die "unsupported architecture '$arch' (prebuilt binaries exist for x86_64 and aarch64; see README for building from source)" ;;
    esac
    case "$os" in
        Darwin)
            # A shell running under Rosetta reports x86_64 on Apple Silicon; install the native build.
            if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
                arch=aarch64
            fi
            TARGET="${arch}-apple-darwin"
            ;;
        Linux)
            if ldd --version 2>&1 | grep -qi musl; then
                die "musl-based Linux (e.g. Alpine) has no prebuilt binary yet; build from source (see README)"
            fi
            TARGET="${arch}-unknown-linux-gnu"
            ;;
        MINGW* | MSYS* | CYGWIN* | Windows_NT)
            die "on Windows, download ${RELEASES}/latest/download/ohmygpu-x86_64-pc-windows-msvc.zip and put ohmygpu-runtime.exe and ohmygpu.exe on your PATH"
            ;;
        *) die "unsupported OS '$os' (prebuilt binaries exist for macOS and Linux)" ;;
    esac
}

# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

pick_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DL=curl
    elif command -v wget >/dev/null 2>&1; then
        DL=wget
    else
        die "need curl or wget to download the release"
    fi
}

# fetch URL DEST [progress]
fetch() {
    case "$DL" in
        curl)
            if [ "${3:-}" = progress ] && [ -t 2 ]; then
                curl -fL --retry 3 --proto '=https' --tlsv1.2 -# -o "$2" "$1"
            else
                curl -fsSL --retry 3 --proto '=https' --tlsv1.2 -o "$2" "$1"
            fi
            ;;
        wget)
            if [ "${3:-}" = progress ] && [ -t 2 ]; then
                wget -q --show-progress --https-only -O "$2" "$1"
            else
                wget -q --https-only -O "$2" "$1"
            fi
            ;;
    esac
}

# The tag "releases/latest" currently points to (empty if it cannot be resolved).
latest_tag() {
    case "$DL" in
        curl)
            loc=$(curl -fsSI --proto '=https' --tlsv1.2 -o /dev/null -w '%{redirect_url}' "$RELEASES/latest" 2>/dev/null || true)
            ;;
        wget)
            loc=$(wget -q -S --max-redirect=0 -O /dev/null "$RELEASES/latest" 2>&1 | sed -n 's/^ *Location: *//p' | tail -n 1 || true)
            ;;
    esac
    printf '%s' "${loc##*/}"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        printf ''
    fi
}

# Follow symlinks to the real file (no `readlink -f` on older macOS).
resolve_link() {
    p="$1"
    n=0
    while [ -L "$p" ] && [ "$n" -lt 20 ]; do
        t=$(readlink "$p")
        case "$t" in
            /*) p="$t" ;;
            *) p="$(dirname "$p")/$t" ;;
        esac
        n=$((n + 1))
    done
    printf '%s' "$p"
}

# "omg 0.3.2" -> "0.3.2"
binary_version() {
    "$1" --version 2>/dev/null | awk '{print $2}' || true
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --version)
                [ $# -ge 2 ] || die "--version needs a value (e.g. v0.5.0)"
                VERSION="$2"
                shift 2
                ;;
            --version=*) VERSION="${1#--version=}"; shift ;;
            --dir)
                [ $# -ge 2 ] || die "--dir needs a value"
                INSTALL_DIR="$2"
                shift 2
                ;;
            --dir=*) INSTALL_DIR="${1#--dir=}"; shift ;;
            --no-sudo) NO_SUDO=1; shift ;;
            --force) FORCE=1; shift ;;
            -h | --help) usage; exit 0 ;;
            *) die "unknown option '$1' (try --help)" ;;
        esac
    done

    detect_target
    pick_downloader

    # Which release?
    if [ "$VERSION" = latest ]; then
        TAG=$(latest_tag)
        if [ -n "$TAG" ]; then
            DL_BASE="$RELEASES/download/$TAG"
        else
            DL_BASE="$RELEASES/latest/download"
        fi
    else
        case "$VERSION" in
            v*) TAG="$VERSION" ;;
            *) TAG="v$VERSION" ;;
        esac
        DL_BASE="$RELEASES/download/$TAG"
    fi

    # Existing install (upgrade in place)?
    existing=$(command -v ohmygpu 2>/dev/null || true)
    existing_version=""
    existing_dir=""
    if [ -n "$existing" ]; then
        existing_version=$(binary_version "$existing")
        existing_dir=$(dirname "$(resolve_link "$existing")")
    fi

    # Where to install.
    if [ -z "$INSTALL_DIR" ]; then
        case "$existing_dir" in
            */Cellar/*) die "ohmygpu is installed with Homebrew ($existing) — upgrade it with: brew upgrade ohmygpu   (or pass --dir to install a separate copy)" ;;
        esac
        if [ -n "$existing_dir" ]; then
            INSTALL_DIR="$existing_dir"
        elif [ -w /usr/local/bin ] || { [ -d /usr/local/bin ] && [ -z "$NO_SUDO" ] && command -v sudo >/dev/null 2>&1; }; then
            INSTALL_DIR=/usr/local/bin
        else
            INSTALL_DIR="$HOME/.local/bin"
        fi
    fi

    if [ -z "$FORCE" ] && [ -n "$TAG" ] && [ -n "$existing_version" ] \
        && [ "v$existing_version" = "$TAG" ] && [ "$existing_dir" = "$INSTALL_DIR" ]; then
        info "OhMyGPU Runtime $TAG is already installed in $INSTALL_DIR — nothing to do (use --force to reinstall)."
        exit 0
    fi

    # Do we need sudo?
    SUDO=""
    if [ ! -d "$INSTALL_DIR" ]; then
        mkdir -p "$INSTALL_DIR" 2>/dev/null || SUDO=sudo
    elif [ ! -w "$INSTALL_DIR" ]; then
        SUDO=sudo
    fi
    if [ -n "$SUDO" ]; then
        [ -z "$NO_SUDO" ] || die "$INSTALL_DIR is not writable; drop --no-sudo or choose a directory with --dir (e.g. --dir ~/.local/bin)"
        command -v sudo >/dev/null 2>&1 || die "$INSTALL_DIR is not writable and sudo is not available; choose a directory with --dir (e.g. --dir ~/.local/bin)"
        info "Installing into $INSTALL_DIR needs sudo — you may be asked for your password."
    fi

    # Download + verify.
    TMP=$(mktemp -d 2>/dev/null || mktemp -d -t ohmygpu)
    trap 'rm -rf "$TMP"' EXIT INT TERM
    asset="ohmygpu-${TARGET}.tar.gz"
    info "Downloading $asset (${TAG:-latest}) …"
    fetch "$DL_BASE/$asset" "$TMP/$asset" progress \
        || die "download failed: $DL_BASE/$asset (no such release, or no build for $TARGET — see $RELEASES)"
    fetch "$DL_BASE/SHA256SUMS.txt" "$TMP/SHA256SUMS.txt" \
        || die "download failed: $DL_BASE/SHA256SUMS.txt"
    expected=$(grep " $asset\$" "$TMP/SHA256SUMS.txt" | awk '{print $1}')
    [ -n "$expected" ] || die "no checksum for $asset in SHA256SUMS.txt"
    actual=$(sha256_of "$TMP/$asset")
    if [ -z "$actual" ]; then
        warn "no sha256sum/shasum/openssl found — skipping checksum verification"
    elif [ "$actual" != "$expected" ]; then
        die "checksum mismatch for $asset (expected $expected, got $actual) — refusing to install"
    fi

    tar xzf "$TMP/$asset" -C "$TMP"
    src="$TMP/ohmygpu-${TARGET}"
    if [ ! -f "$src/ohmygpu" ] || [ ! -f "$src/ohmygpu-runtime" ]; then
        die "unexpected archive layout in $asset (expected $src/ohmygpu and $src/ohmygpu-runtime)"
    fi

    # Install. `install` replaces the files (unlink + create), so a running runtime is unaffected.
    $SUDO mkdir -p "$INSTALL_DIR"
    $SUDO install -m 755 "$src/ohmygpu-runtime" "$src/ohmygpu" "$INSTALL_DIR/"
    files="ohmygpu-runtime, ohmygpu"
    if [ -e "$INSTALL_DIR/omg" ] && [ ! -L "$INSTALL_DIR/omg" ]; then
        warn "$INSTALL_DIR/omg exists and is not a symlink — leaving it alone (use 'ohmygpu', or alias omg=ohmygpu)"
    else
        $SUDO ln -sf ohmygpu "$INSTALL_DIR/omg"
        files="$files, omg"
    fi

    new_version=$(binary_version "$INSTALL_DIR/ohmygpu")
    if [ -z "$new_version" ]; then
        die "$INSTALL_DIR/ohmygpu was installed but does not run — on macOS try: xattr -dr com.apple.quarantine $INSTALL_DIR/ohmygpu $INSTALL_DIR/ohmygpu-runtime"
    fi

    info ""
    if [ -n "$existing_version" ] && [ "$existing_dir" = "$INSTALL_DIR" ]; then
        info "Upgraded OhMyGPU Runtime v$existing_version → v$new_version in $INSTALL_DIR"
    else
        info "Installed OhMyGPU Runtime v$new_version to $INSTALL_DIR ($files)"
    fi

    # PATH / shadowing hints.
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) on_path=1 ;;
        *) on_path="" ;;
    esac
    if [ -z "$on_path" ]; then
        shown_dir="$INSTALL_DIR"
        case "$INSTALL_DIR" in "$HOME"/*) shown_dir="\$HOME${INSTALL_DIR#"$HOME"}" ;; esac
        info ""
        info "$INSTALL_DIR is not on your PATH. Add it:"
        case "$(basename "${SHELL:-sh}")" in
            fish) info "  fish_add_path $shown_dir" ;;
            zsh) info "  echo 'export PATH=\"$shown_dir:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
            bash)
                if [ "$(uname -s)" = Darwin ]; then
                    info "  echo 'export PATH=\"$shown_dir:\$PATH\"' >> ~/.bash_profile && source ~/.bash_profile"
                else
                    info "  echo 'export PATH=\"$shown_dir:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
                fi
                ;;
            *) info "  export PATH=\"$shown_dir:\$PATH\"" ;;
        esac
    else
        found=$(command -v ohmygpu 2>/dev/null || true)
        if [ -n "$found" ] && [ "$(resolve_link "$found")" != "$(resolve_link "$INSTALL_DIR/ohmygpu")" ]; then
            warn "another ohmygpu at $found comes first on your PATH and will shadow $INSTALL_DIR/ohmygpu"
        fi
    fi

    # A runtime that is already running keeps the old version until restarted.
    if [ "$DL" = curl ]; then
        port="${OHMYGPU_PORT:-10692}"
        health=$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/ohmygpu/v1/health" 2>/dev/null || true)
        case "$health" in
            *'"ok"'*)
                info ""
                info "A runtime is running on port $port; restart it to use v$new_version:  omg shutdown && omg serve"
                ;;
        esac
    fi

    info ""
    info "Next:"
    info "  omg serve                               # start the runtime (foreground)"
    info "  omg model pull qwen2.5-0.5b-instruct    # download a model (omg model catalog lists more)"
    info "  omg run qwen2.5-0.5b-instruct           # start it, then POST http://127.0.0.1:10692/v1/responses"
    info ""
    info "Upgrade later with:  omg upgrade   (or run this script again)"
}

main "$@"
