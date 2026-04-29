use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct ProgressBar {
    enabled: bool,
    label: String,
    total: usize,
    current: AtomicUsize,
}

impl ProgressBar {
    pub fn new(label: &str, total: usize, enabled: bool) -> Self {
        Self {
            enabled: enabled && total > 0,
            label: label.to_string(),
            total,
            current: AtomicUsize::new(0),
        }
    }

    pub fn tick(&self, message: impl AsRef<str>) {
        if !self.enabled {
            return;
        }
        let current = (self.current.fetch_add(1, Ordering::Relaxed) + 1).min(self.total);
        eprint!(
            "\r{} [{}/{}] {}",
            self.label,
            current,
            self.total,
            message.as_ref()
        );
        let _ = io::stderr().flush();
    }

    pub fn finish(&self) {
        if self.enabled {
            eprintln!();
        }
    }
}
