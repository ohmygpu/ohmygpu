use anyhow::Result;
use ohmygpu_core::Config;

pub async fn execute(key: Option<&str>, value: Option<&str>) -> Result<()> {
    let mut config = Config::load()?;

    match (key, value) {
        // Show all config
        (None, None) => {
            println!("Configuration file: {:?}\n", Config::config_path()?);
            println!("[daemon]");
            println!("  host = \"{}\"", config.daemon.host);
            println!("  port = {}", config.daemon.port);
            println!();
            println!("[models]");
            println!(
                "  directory = {:?}",
                config.models.directory.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(default)".to_string())
            );
            println!(
                "  hf_token = {}",
                config.models.hf_token.as_ref().map(|_| "***").unwrap_or("(not set)")
            );
            println!();
            println!("[inference]");
            println!("  max_tokens = {}", config.inference.max_tokens);
            println!("  temperature = {}", config.inference.temperature);
            println!("  top_p = {}", config.inference.top_p);
            println!("  use_gpu = {}", config.inference.use_gpu);
        }

        // Get a specific key
        (Some(key), None) => {
            let value = get_config_value(&config, key)?;
            println!("{}", value);
        }

        // Set a specific key
        (Some(key), Some(value)) => {
            set_config_value(&mut config, key, value)?;
            config.save()?;
            println!("Set {} = {}", key, value);
        }

        _ => unreachable!(),
    }

    Ok(())
}

fn get_config_value(config: &Config, key: &str) -> Result<String> {
    match key {
        "daemon.host" => Ok(config.daemon.host.clone()),
        "daemon.port" => Ok(config.daemon.port.to_string()),
        "models.directory" => Ok(config
            .models
            .directory
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()),
        "models.hf_token" => Ok(config
            .models
            .hf_token
            .as_ref()
            .map(|_| "***".to_string())
            .unwrap_or_default()),
        "inference.max_tokens" => Ok(config.inference.max_tokens.to_string()),
        "inference.temperature" => Ok(config.inference.temperature.to_string()),
        "inference.top_p" => Ok(config.inference.top_p.to_string()),
        "inference.use_gpu" => Ok(config.inference.use_gpu.to_string()),
        _ => anyhow::bail!("Unknown config key: {}", key),
    }
}

fn set_config_value(config: &mut Config, key: &str, value: &str) -> Result<()> {
    match key {
        "daemon.host" => config.daemon.host = value.to_string(),
        "daemon.port" => config.daemon.port = value.parse()?,
        "models.directory" => {
            config.models.directory = if value.is_empty() {
                None
            } else {
                Some(value.into())
            }
        }
        "models.hf_token" => {
            config.models.hf_token = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        "inference.max_tokens" => config.inference.max_tokens = value.parse()?,
        "inference.temperature" => config.inference.temperature = value.parse()?,
        "inference.top_p" => config.inference.top_p = value.parse()?,
        "inference.use_gpu" => config.inference.use_gpu = value.parse()?,
        _ => anyhow::bail!("Unknown config key: {}", key),
    }
    Ok(())
}
