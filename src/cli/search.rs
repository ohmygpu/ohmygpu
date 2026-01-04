use anyhow::Result;

use crate::downloaders::huggingface::HuggingFaceDownloader;

pub async fn execute(query: &str) -> Result<()> {
    let downloader = HuggingFaceDownloader::new();
    let results = downloader.search(query).await?;

    if results.is_empty() {
        println!("No models found for '{}'", query);
        return Ok(());
    }

    println!("{:<50} {:<15} {:<10}", "MODEL", "DOWNLOADS", "LIKES");
    println!("{}", "-".repeat(75));

    for model in results.iter().take(20) {
        println!(
            "{:<50} {:<15} {:<10}",
            truncate(&model.id, 48),
            format_number(model.downloads),
            model.likes
        );
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
