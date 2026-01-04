mod app;
mod ui;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;

use app::{App, AppMode};

pub async fn run() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new().await?;

    // Main loop
    let result = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match app.mode {
                    AppMode::Normal => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('j') | KeyCode::Down => app.next(),
                        KeyCode::Char('k') | KeyCode::Up => app.previous(),
                        KeyCode::Char('s') => app.mode = AppMode::Search,
                        KeyCode::Enter => app.select().await?,
                        KeyCode::Char('d') => app.delete_selected().await?,
                        KeyCode::Tab => app.next_tab(),
                        KeyCode::BackTab => app.previous_tab(),
                        _ => {}
                    },
                    AppMode::Search => match key.code {
                        KeyCode::Esc => {
                            app.search_query.clear();
                            app.mode = AppMode::Normal;
                        }
                        KeyCode::Enter => {
                            app.perform_search().await?;
                            app.mode = AppMode::Normal;
                        }
                        KeyCode::Backspace => {
                            app.search_query.pop();
                        }
                        KeyCode::Char(c) => {
                            app.search_query.push(c);
                        }
                        _ => {}
                    },
                    AppMode::Downloading => {
                        // Can't interrupt downloads for now
                        if key.code == KeyCode::Esc {
                            app.mode = AppMode::Normal;
                        }
                    }
                }
            }
        }

        // Check for async updates
        app.tick().await?;
    }
}
