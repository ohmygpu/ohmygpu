use anyhow::Result;

use crate::downloaders::huggingface::{HfSearchResult, HuggingFaceDownloader};
use crate::models::registry::ModelRegistry;
use crate::models::ModelInfo;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Normal,
    Search,
    Downloading,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tab {
    Models,
    Search,
    Settings,
}

pub struct App {
    pub mode: AppMode,
    pub current_tab: Tab,

    // Models tab
    pub models: Vec<ModelInfo>,
    pub selected_model: usize,

    // Search tab
    pub search_query: String,
    pub search_results: Vec<HfSearchResult>,
    pub selected_result: usize,

    // Status
    pub status_message: String,
    pub is_loading: bool,

    // Internal
    registry: ModelRegistry,
    downloader: HuggingFaceDownloader,
}

impl App {
    pub async fn new() -> Result<Self> {
        let registry = ModelRegistry::load()?;
        let models: Vec<ModelInfo> = registry.list().into_iter().cloned().collect();

        Ok(Self {
            mode: AppMode::Normal,
            current_tab: Tab::Models,
            models,
            selected_model: 0,
            search_query: String::new(),
            search_results: Vec::new(),
            selected_result: 0,
            status_message: String::from("Press 's' to search, 'q' to quit"),
            is_loading: false,
            registry,
            downloader: HuggingFaceDownloader::new(),
        })
    }

    pub fn next(&mut self) {
        match self.current_tab {
            Tab::Models => {
                if !self.models.is_empty() {
                    self.selected_model = (self.selected_model + 1) % self.models.len();
                }
            }
            Tab::Search => {
                if !self.search_results.is_empty() {
                    self.selected_result = (self.selected_result + 1) % self.search_results.len();
                }
            }
            Tab::Settings => {}
        }
    }

    pub fn previous(&mut self) {
        match self.current_tab {
            Tab::Models => {
                if !self.models.is_empty() {
                    self.selected_model = self
                        .selected_model
                        .checked_sub(1)
                        .unwrap_or(self.models.len() - 1);
                }
            }
            Tab::Search => {
                if !self.search_results.is_empty() {
                    self.selected_result = self
                        .selected_result
                        .checked_sub(1)
                        .unwrap_or(self.search_results.len() - 1);
                }
            }
            Tab::Settings => {}
        }
    }

    pub fn next_tab(&mut self) {
        self.current_tab = match self.current_tab {
            Tab::Models => Tab::Search,
            Tab::Search => Tab::Settings,
            Tab::Settings => Tab::Models,
        };
    }

    pub fn previous_tab(&mut self) {
        self.current_tab = match self.current_tab {
            Tab::Models => Tab::Settings,
            Tab::Search => Tab::Models,
            Tab::Settings => Tab::Search,
        };
    }

    pub async fn select(&mut self) -> Result<()> {
        match self.current_tab {
            Tab::Models => {
                if let Some(model) = self.models.get(self.selected_model) {
                    self.status_message = format!("Selected: {}", model.name);
                    // TODO: Run the model
                }
            }
            Tab::Search => {
                if let Some(result) = self.search_results.get(self.selected_result) {
                    self.status_message = format!("Downloading: {}", result.id);
                    self.mode = AppMode::Downloading;
                    self.is_loading = true;

                    // Download in background
                    let model_id = result.id.clone();
                    match self.download_model(&model_id).await {
                        Ok(model_info) => {
                            self.models.push(model_info);
                            self.status_message = format!("Downloaded: {}", model_id);
                        }
                        Err(e) => {
                            self.status_message = format!("Error: {}", e);
                        }
                    }

                    self.is_loading = false;
                    self.mode = AppMode::Normal;
                }
            }
            Tab::Settings => {}
        }
        Ok(())
    }

    pub async fn delete_selected(&mut self) -> Result<()> {
        if self.current_tab == Tab::Models && !self.models.is_empty() {
            let model = self.models.remove(self.selected_model);
            self.registry.remove(&model.name)?;
            self.registry.save()?;

            if let Err(e) = std::fs::remove_dir_all(&model.path) {
                self.status_message = format!("Warning: Could not delete files: {}", e);
            } else {
                self.status_message = format!("Deleted: {}", model.name);
            }

            if self.selected_model >= self.models.len() && self.selected_model > 0 {
                self.selected_model -= 1;
            }
        }
        Ok(())
    }

    pub async fn perform_search(&mut self) -> Result<()> {
        if self.search_query.is_empty() {
            return Ok(());
        }

        self.is_loading = true;
        self.status_message = format!("Searching for '{}'...", self.search_query);
        self.current_tab = Tab::Search;

        match self.downloader.search(&self.search_query).await {
            Ok(results) => {
                self.search_results = results;
                self.selected_result = 0;
                self.status_message = format!(
                    "Found {} results for '{}'",
                    self.search_results.len(),
                    self.search_query
                );
            }
            Err(e) => {
                self.status_message = format!("Search failed: {}", e);
            }
        }

        self.is_loading = false;
        self.search_query.clear();
        Ok(())
    }

    async fn download_model(&mut self, model_id: &str) -> Result<ModelInfo> {
        use crate::downloaders::Downloader;
        let model_info = self.downloader.download(model_id, None).await?;
        self.registry.add(model_info.clone())?;
        self.registry.save()?;
        Ok(model_info)
    }

    pub async fn tick(&mut self) -> Result<()> {
        // Handle background tasks, refresh state, etc.
        Ok(())
    }
}
