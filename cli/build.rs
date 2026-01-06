fn main() {
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    compile_error!(
        "ohmygpu requires GPU acceleration. Build with either:\n\
         \n\
         - cargo build --release --features metal   (Apple Silicon)\n\
         - cargo build --release --features cuda    (NVIDIA GPU)"
    );
}
