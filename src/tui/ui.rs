use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs},
    Frame,
};

use super::app::{App, AppMode, Tab};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Status bar
        ])
        .split(f.area());

    draw_tabs(f, app, chunks[0]);
    draw_content(f, app, chunks[1]);
    draw_status_bar(f, app, chunks[2]);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles = vec!["Models", "Search", "Settings"];
    let selected = match app.current_tab {
        Tab::Models => 0,
        Tab::Search => 1,
        Tab::Settings => 2,
    };

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" omg - ohmygpu "),
        )
        .select(selected)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn draw_content(f: &mut Frame, app: &App, area: Rect) {
    match app.current_tab {
        Tab::Models => draw_models_tab(f, app, area),
        Tab::Search => draw_search_tab(f, app, area),
        Tab::Settings => draw_settings_tab(f, app, area),
    }
}

fn draw_models_tab(f: &mut Frame, app: &App, area: Rect) {
    if app.models.is_empty() {
        let message = Paragraph::new(vec![
            Line::from(""),
            Line::from("No models downloaded yet.").centered(),
            Line::from("").centered(),
            Line::from("Press 's' to search and download models.").centered(),
            Line::from("Or use: omg pull <model-name>").centered(),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Models "));
        f.render_widget(message, area);
        return;
    }

    let items: Vec<ListItem> = app
        .models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let style = if i == app.selected_model {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let size = format_size(model.size_bytes);
            let content = Line::from(vec![
                Span::styled(
                    if i == app.selected_model {
                        "> "
                    } else {
                        "  "
                    },
                    style,
                ),
                Span::styled(&model.name, style),
                Span::raw("  "),
                Span::styled(model.model_type.as_str(), Style::default().fg(Color::DarkGray)),
                Span::raw("  "),
                Span::styled(size, Style::default().fg(Color::Green)),
            ]);

            ListItem::new(content)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Models (Enter: run, d: delete) "),
    );

    f.render_widget(list, area);
}

fn draw_search_tab(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Search input
    let search_style = if app.mode == AppMode::Search {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let search_text = if app.mode == AppMode::Search {
        format!("{}▋", app.search_query)
    } else if app.search_query.is_empty() {
        "Press 's' to search...".to_string()
    } else {
        app.search_query.clone()
    };

    let search_input = Paragraph::new(search_text)
        .style(search_style)
        .block(Block::default().borders(Borders::ALL).title(" Search "));

    f.render_widget(search_input, chunks[0]);

    // Results
    if app.search_results.is_empty() {
        let message = Paragraph::new("Search for models on HuggingFace")
            .centered()
            .block(Block::default().borders(Borders::ALL).title(" Results "));
        f.render_widget(message, chunks[1]);
        return;
    }

    let items: Vec<ListItem> = app
        .search_results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let style = if i == app.selected_result {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let downloads = format_number(result.downloads);
            let content = Line::from(vec![
                Span::styled(
                    if i == app.selected_result {
                        "> "
                    } else {
                        "  "
                    },
                    style,
                ),
                Span::styled(&result.id, style),
                Span::raw("  "),
                Span::styled(
                    format!("↓{}", downloads),
                    Style::default().fg(Color::Green),
                ),
                Span::raw("  "),
                Span::styled(format!("♥{}", result.likes), Style::default().fg(Color::Red)),
            ]);

            ListItem::new(content)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Results (Enter: download) "),
    );

    f.render_widget(list, chunks[1]);
}

fn draw_settings_tab(f: &mut Frame, _app: &App, area: Rect) {
    let settings = Paragraph::new(vec![
        Line::from(""),
        Line::from("  Settings coming soon..."),
        Line::from(""),
        Line::from("  • Model storage location"),
        Line::from("  • Default quantization"),
        Line::from("  • GPU/CPU preferences"),
        Line::from("  • HuggingFace token"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Settings "));

    f.render_widget(settings, area);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let status_style = if app.is_loading {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let loading_indicator = if app.is_loading { "⏳ " } else { "" };

    let help_text = match app.mode {
        AppMode::Normal => " q: quit | s: search | Tab: switch tabs | j/k: navigate | Enter: select ",
        AppMode::Search => " Enter: search | Esc: cancel ",
        AppMode::Downloading => " Downloading... ",
    };

    let status = Paragraph::new(Line::from(vec![
        Span::styled(loading_indicator, Style::default().fg(Color::Yellow)),
        Span::styled(&app.status_message, status_style),
        Span::raw("  "),
        Span::styled(help_text, Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL));

    f.render_widget(status, area);
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
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
