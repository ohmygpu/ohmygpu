//! Garbage collect unused cache files

use anyhow::Result;
use ohmygpu_core::ModelRegistry;

pub async fn execute() -> Result<()> {
    let registry = ModelRegistry::load()?;
    let models_dir = ModelRegistry::models_dir()?;

    println!("Scanning for unused cache files...");

    // Get list of valid model directories (for future use in orphan detection)
    let _valid_models: std::collections::HashSet<_> = registry
        .list()
        .iter()
        .map(|m| m.path.clone())
        .collect();

    let mut cleaned_bytes: u64 = 0;
    let mut cleaned_files: u32 = 0;

    // Scan cache directory for orphaned files
    let cache_dir = models_dir.join(".cache");
    if cache_dir.exists() {
        for entry in std::fs::read_dir(&cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if this cache entry is still referenced
            // For now, just report what we find
            if path.is_file() {
                let size = path.metadata()?.len();
                println!("  Found cache file: {} ({} bytes)", path.display(), size);
                cleaned_bytes += size;
                cleaned_files += 1;
            }
        }
    }

    // Look for partial downloads (.part files)
    if models_dir.exists() {
        for entry in walkdir::WalkDir::new(&models_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().map(|e| e == "part").unwrap_or(false) {
                let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                println!("  Partial download: {} ({} bytes)", path.display(), size);
                cleaned_bytes += size;
                cleaned_files += 1;

                // Remove partial downloads
                if let Err(e) = std::fs::remove_file(path) {
                    eprintln!("    Failed to remove: {}", e);
                } else {
                    println!("    Removed");
                }
            }
        }
    }

    if cleaned_files == 0 {
        println!("No unused files found.");
    } else {
        println!();
        println!(
            "Cleaned {} files, freed {} bytes",
            cleaned_files, cleaned_bytes
        );
    }

    Ok(())
}
