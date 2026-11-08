use crate::label::Label;

pub fn create() -> Box<dyn Label> {
    super::create_open_return(10)
}
