use std::error::Error;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
struct SimpleError(String);

impl Display for SimpleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SimpleError {}

pub fn err(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    Box::new(SimpleError(message.into()))
}
