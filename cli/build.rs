fn main() {
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        eprintln!("error: ohmygpu requires GPU acceleration. Build with either:");
        eprintln!();
        eprintln!("  make build-metal   (Apple Silicon)");
        eprintln!("  make build-cuda    (NVIDIA GPU)");
        std::process::exit(1);
    }
}
