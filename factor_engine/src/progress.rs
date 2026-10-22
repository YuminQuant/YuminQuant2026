use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct ProgressBar {
    enabled: bool,
    label: String,
    total: usize,
    current: AtomicUsize,
    line_width: AtomicUsize,
}

impl ProgressBar {
    pub fn new(label: &str, total: usize, enabled: bool) -> Self {
        Self {
            enabled: enabled && total > 0,
            label: label.to_string(),
            total,
            current: AtomicUsize::new(0),
            line_width: AtomicUsize::new(0),
        }
    }

    pub fn tick(&self, message: impl AsRef<str>) {
        if !self.enabled {
            return;
        }
        self.current.fetch_add(1, Ordering::Relaxed);
        self.update(message);
    }

    /// Report in-flight work without counting another completed batch.
    pub fn update(&self, message: impl AsRef<str>) {
        if !self.enabled {
            return;
        }
        let mut stderr = io::stderr().lock();
        let line = format!(
            "{} [{}/{}] {}",
            self.label,
            self.current.load(Ordering::Relaxed).min(self.total),
            self.total,
            message.as_ref()
        );
        if stderr.is_terminal() {
            let width = line.chars().count();
            let previous = self.line_width.swap(width, Ordering::Relaxed);
            let _ = write!(
                stderr,
                "\r{}{}",
                line,
                " ".repeat(previous.saturating_sub(width))
            );
        } else {
            let _ = writeln!(stderr, "{}", line);
        }
        let _ = stderr.flush();
    }

    pub fn finish(&self) {
        if self.enabled {
            eprintln!();
        }
    }
}
