use anyhow::Result;
use ohmygpu_core::downloaders::HuggingFaceDownloader;

pub async fn execute(query: &str) -> Result<()> {
    println!("Searching HuggingFace for: {}\n", query);

    let downloader = HuggingFaceDownloader::new();
    let results = downloader.search(query).await?;

    if results.is_empty() {
        println!("No models found matching '{}'", query);
        return Ok(());
    }

    println!("{:<50} {:>12} {:>8}", "MODEL", "DOWNLOADS", "LIKES");
    println!("{}", "-".repeat(72));

    for model in results {
        let downloads = format_number(model.downloads);
        println!("{:<50} {:>12} {:>8}", model.id, downloads, model.likes);
    }

    println!("\nRun `omg pull <model>` to download a model.");

    Ok(())
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
