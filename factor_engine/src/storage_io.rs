use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;
use bytes::Bytes;
use parquet::file::reader::{ChunkReader, Length};
use crate::error::{err, Result};

pub trait ArtifactGuard: Send {}
impl<T: Send> ArtifactGuard for T {}

#[derive(Clone, Debug)]
pub struct FileInfo { pub size: u64, pub modified_ns: u128, pub directory: bool }
impl FileInfo {
    pub fn len(&self) -> u64 { self.size }
    pub fn is_file(&self) -> bool { !self.directory }
    pub fn is_dir(&self) -> bool { self.directory }
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(UNIX_EPOCH + std::time::Duration::from_nanos(self.modified_ns.min(u64::MAX as u128) as u64))
    }
}
pub trait ReadAt: Send + Sync {
    fn len(&self) -> u64;
    fn read_at(&self, offset: u64, length: usize) -> io::Result<Bytes>;
}
pub trait AtomicWrite: Write + Send { fn commit(self: Box<Self>) -> Result<()>; }
pub trait ArtifactBackend: Send + Sync {
    fn handles(&self, path: &Path) -> bool;
    fn stat(&self, path: &Path) -> Result<Option<FileInfo>>;
    fn list(&self, path: &Path) -> Result<Vec<PathBuf>>;
    fn create_dir_all(&self, path: &Path) -> Result<()>;
    fn open_read(&self, path: &Path) -> Result<Arc<dyn ReadAt>>;
    fn open_write(&self, path: &Path) -> Result<Box<dyn AtomicWrite>>;
    fn remove_file(&self, path: &Path) -> Result<()>;
    fn rename(&self, from: &Path, to: &Path) -> Result<()>;
    fn lock(&self, path: &Path) -> Result<Box<dyn ArtifactGuard>>;
    fn exists(&self, path: &Path) -> Result<bool> { Ok(self.stat(path)?.is_some()) }
}
static BACKEND: OnceLock<Arc<dyn ArtifactBackend>> = OnceLock::new();
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
static OPEN_FILES: AtomicU64 = AtomicU64::new(0);
static STAT_CALLS: AtomicU64 = AtomicU64::new(0);
static LIST_CALLS: AtomicU64 = AtomicU64::new(0);
static READ_NS: AtomicU64 = AtomicU64::new(0);
static WRITE_BYTES: AtomicU64 = AtomicU64::new(0);
static WRITE_NS: AtomicU64 = AtomicU64::new(0);
static OPEN_NS: AtomicU64 = AtomicU64::new(0);
static READ_RANGES: AtomicU64 = AtomicU64::new(0);
static READ_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub fn profile() -> String {
    format!("storage read_bytes={} opens={} stats={} lists={} read_ms={} write_bytes={} open_ms={} write_ms={} read_ranges={} read_cache_hits={}",
        READ_BYTES.load(Ordering::Relaxed), OPEN_FILES.load(Ordering::Relaxed),
        STAT_CALLS.load(Ordering::Relaxed), LIST_CALLS.load(Ordering::Relaxed),
        READ_NS.load(Ordering::Relaxed)/1_000_000, WRITE_BYTES.load(Ordering::Relaxed),
        OPEN_NS.load(Ordering::Relaxed)/1_000_000, WRITE_NS.load(Ordering::Relaxed)/1_000_000,
        READ_RANGES.load(Ordering::Relaxed), READ_CACHE_HITS.load(Ordering::Relaxed))
}
pub fn register(backend: Arc<dyn ArtifactBackend>) -> Result<()> {
    BACKEND.set(backend).map_err(|_| err("storage backend already registered"))
}
pub fn backend(path: &Path) -> Option<&'static dyn ArtifactBackend> {
    BACKEND.get().map(Arc::as_ref).filter(|b| b.handles(path))
}
fn io_error(error: impl std::fmt::Display) -> io::Error { io::Error::other(error.to_string()) }
fn snapshot_path(path: &Path) -> PathBuf {
    std::env::var("YQ_METADATA_SNAPSHOTS").ok().and_then(|mapping| {
        mapping.lines().find_map(|line| {
            let (source, target) = line.split_once('\t')?;
            (path == Path::new(source)).then(|| PathBuf::from(target))
        })
    }).unwrap_or_else(||path.to_owned())
}
pub fn metadata(path: impl AsRef<Path>) -> io::Result<FileInfo> {
    let resolved = snapshot_path(path.as_ref());
    let path = resolved.as_path();
    STAT_CALLS.fetch_add(1, Ordering::Relaxed);
    if let Some(b) = backend(path) {
        return b.stat(path).map_err(io_error)?.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, path.display().to_string()));
    }
    let info = std::fs::metadata(path)?;
    Ok(FileInfo { size: info.len(), modified_ns: info.modified()?.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(), directory: info.is_dir() })
}
pub fn exists(path: &Path) -> Result<bool> {
    match metadata(path) {
        Ok(_) => Ok(true), Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false), Err(e) => Err(e.into()),
    }
}
/// Retains the existing catalog's boolean API, but never hides remote IO errors.
pub trait StoragePathExt {
    fn storage_exists(&self) -> bool;
    fn storage_is_file(&self) -> bool;
    fn storage_is_dir(&self) -> bool;
}
impl<T: AsRef<Path>> StoragePathExt for T {
    fn storage_exists(&self) -> bool {
        exists(self.as_ref()).unwrap_or_else(|e| panic!("storage lookup failed for {}: {e}", self.as_ref().display()))
    }
    fn storage_is_file(&self) -> bool { self.storage_exists() && metadata(self).expect("storage stat").is_file() }
    fn storage_is_dir(&self) -> bool { self.storage_exists() && metadata(self).expect("storage stat").is_dir() }
}
pub struct DirEntry { path: PathBuf }
impl DirEntry {
    pub fn path(&self) -> PathBuf { self.path.clone() }
    pub fn file_name(&self) -> std::ffi::OsString { self.path.file_name().unwrap_or_default().into() }
}
pub fn read_dir(path: impl AsRef<Path>) -> io::Result<std::vec::IntoIter<io::Result<DirEntry>>> {
    let path = path.as_ref();
    LIST_CALLS.fetch_add(1, Ordering::Relaxed);
    let names = if let Some(b) = backend(path) { b.list(path).map_err(io_error)? }
        else { std::fs::read_dir(path)?.map(|entry| entry.map(|e| e.path())).collect::<io::Result<Vec<_>>>()? };
    Ok(names.into_iter().map(|path| Ok(DirEntry { path })).collect::<Vec<_>>().into_iter())
}
pub fn create_dir_all(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    match backend(path) { Some(b) => b.create_dir_all(path), None => Ok(std::fs::create_dir_all(path)?) }
}
pub fn remove_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    match backend(path) { Some(b) => b.remove_file(path), None => Ok(std::fs::remove_file(path)?) }
}
pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    let (from,to) = (from.as_ref(),to.as_ref());
    match (backend(from),backend(to)) {
        (Some(b),Some(_)) => b.rename(from,to), (None,None) => Ok(std::fs::rename(from,to)?),
        _ => Err(err("cross-backend rename is forbidden")),
    }
}
pub fn lock(path: &Path) -> Result<Option<Box<dyn ArtifactGuard>>> {
    if let Some(b) = backend(path) { return Ok(Some(b.lock(path)?)); }
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let name = if path.extension().and_then(|s|s.to_str()) == Some("lock") {path.to_owned()}
        else {PathBuf::from(format!("{}.lock",path.display()))};
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(name)?;
    if file.metadata()?.len() == 0 { file.set_len(1)?; }
    file.lock()?;
    Ok(Some(Box::new(file)))
}
struct LocalRead { file: Mutex<std::fs::File>, len: u64 }
impl ReadAt for LocalRead {
    fn len(&self) -> u64 { self.len }
    fn read_at(&self, offset: u64, length: usize) -> io::Result<Bytes> {
        let mut file = self.file.lock().map_err(io_error)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0;length]; file.read_exact(&mut bytes)?; Ok(bytes.into())
    }
}
#[derive(Clone)]
pub struct StorageReader(Arc<BufferedRead>);
const READ_BLOCK_BYTES: usize = 64 * 1024;
const READ_CACHE_BLOCKS: usize = 16;
struct BufferedRead {
    source: Arc<dyn ReadAt>,
    blocks: Mutex<VecDeque<(u64, Bytes)>>,
}
impl Length for StorageReader { fn len(&self) -> u64 { self.0.source.len() } }
impl StorageReader {
    fn new(source: Arc<dyn ReadAt>) -> Self {
        Self(Arc::new(BufferedRead { source, blocks: Mutex::new(VecDeque::new()) }))
    }

    fn range(&self, offset: u64, length: usize) -> io::Result<Bytes> {
        if offset > self.len() || length as u64 > self.len() - offset {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read range exceeds file length"));
        }
        if length == 0 { return Ok(Bytes::new()); }
        let mut blocks = self.0.blocks.lock().map_err(io_error)?;
        let mut cursor = offset;
        let end = offset + length as u64;
        let mut output = Vec::new();
        while cursor < end {
            let start = cursor / READ_BLOCK_BYTES as u64 * READ_BLOCK_BYTES as u64;
            let bytes = if let Some(index) = blocks.iter().position(|(at, _)| *at == start) {
                READ_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                let entry = blocks.remove(index).expect("located cache block");
                let bytes = entry.1.clone();
                blocks.push_back(entry);
                bytes
            } else {
                let needed = (end - start).div_ceil(READ_BLOCK_BYTES as u64)
                    .min(READ_CACHE_BLOCKS as u64) * READ_BLOCK_BYTES as u64;
                let size = (self.len() - start).min(needed) as usize;
                let started = std::time::Instant::now();
                let bytes = self.0.source.read_at(start, size)?;
                if bytes.len() != size {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short storage read"));
                }
                READ_RANGES.fetch_add(1, Ordering::Relaxed);
                READ_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                READ_NS.fetch_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
                for (index, block) in bytes.chunks(READ_BLOCK_BYTES).enumerate() {
                    if blocks.len() == READ_CACHE_BLOCKS { blocks.pop_front(); }
                    let from = index * READ_BLOCK_BYTES;
                    blocks.push_back((start + from as u64, Bytes::copy_from_slice(block)));
                }
                bytes.slice(..bytes.len().min(READ_BLOCK_BYTES))
            };
            let from = (cursor - start) as usize;
            let count = (end - cursor).min((bytes.len() - from) as u64) as usize;
            if cursor == offset && count == length { return Ok(bytes.slice(from..from + count)); }
            if output.is_empty() { output.reserve(length); }
            output.extend_from_slice(&bytes[from..from + count]);
            cursor += count as u64;
        }
        Ok(output.into())
    }
}
pub struct ReadCursor { reader: StorageReader, offset: u64, buffer: Bytes, position: usize }
impl Read for ReadCursor {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.offset >= self.reader.len() { return Ok(0); }
        if self.position == self.buffer.len() {
            let remaining = READ_BLOCK_BYTES - self.offset as usize % READ_BLOCK_BYTES;
            let length = (self.reader.len()-self.offset).min(remaining as u64) as usize;
            self.buffer = self.reader.range(self.offset,length)?; self.position = 0;
        }
        let count = out.len().min(self.buffer.len()-self.position);
        out[..count].copy_from_slice(&self.buffer[self.position..self.position+count]);
        self.position += count; self.offset += count as u64; Ok(count)
    }
}
impl ChunkReader for StorageReader {
    type T = ReadCursor;
    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        Ok(ReadCursor { reader:self.clone(), offset:start, buffer:Bytes::new(), position:0 })
    }
    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> { Ok(self.range(start,length)?) }
}
pub fn open_reader(path: &Path) -> Result<StorageReader> {
    let resolved = snapshot_path(path);
    let path = resolved.as_path();
    OPEN_FILES.fetch_add(1, Ordering::Relaxed);
    let started = std::time::Instant::now();
    if let Some(b) = backend(path) {
        let reader = StorageReader::new(b.open_read(path)?);
        OPEN_NS.fetch_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64,Ordering::Relaxed);
        return Ok(reader);
    }
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    OPEN_NS.fetch_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64,Ordering::Relaxed);
    Ok(StorageReader::new(Arc::new(LocalRead { file:Mutex::new(file), len })))
}
pub fn read(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let reader = open_reader(path.as_ref())?;
    Ok(reader.range(0,usize::try_from(reader.len())?)?.to_vec())
}
pub fn read_to_string(path: impl AsRef<Path>) -> Result<String> { Ok(String::from_utf8(read(path)?)?) }
struct LocalWrite { file: Option<std::fs::File>, temp: PathBuf, path: PathBuf }
impl Write for LocalWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.file.as_mut().unwrap().write(bytes) }
    fn flush(&mut self) -> io::Result<()> { self.file.as_mut().unwrap().flush() }
}
impl AtomicWrite for LocalWrite {
    fn commit(mut self: Box<Self>) -> Result<()> {
        self.file.take().unwrap().sync_all()?;
        std::fs::rename(&self.temp,&self.path)?; Ok(())
    }
}
impl Drop for LocalWrite {
    fn drop(&mut self) { drop(self.file.take()); let _ = std::fs::remove_file(&self.temp); }
}
pub struct AtomicWriter(Option<Box<dyn AtomicWrite>>);
impl Write for AtomicWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let started = std::time::Instant::now();
        let size = self.0.as_mut().unwrap().write(bytes)?;
        WRITE_NS.fetch_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64,Ordering::Relaxed);
        WRITE_BYTES.fetch_add(size as u64, Ordering::Relaxed); Ok(size)
    }
    fn flush(&mut self) -> io::Result<()> { self.0.as_mut().unwrap().flush() }
}
impl AtomicWriter {
    pub fn commit(mut self) -> Result<()> {
        let started = std::time::Instant::now();
        let result=self.0.take().unwrap().commit();
        WRITE_NS.fetch_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64,Ordering::Relaxed);
        result
    }
}
pub fn open_writer(path: &Path) -> Result<AtomicWriter> {
    if let Some(b) = backend(path) { return Ok(AtomicWriter(Some(b.open_write(path)?))); }
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let temp = path.with_extension(format!("{}.{}.{}.part", std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(), SEQUENCE.fetch_add(1,Ordering::Relaxed)));
    let file = std::fs::File::create_new(&temp)?;
    Ok(AtomicWriter(Some(Box::new(LocalWrite { file:Some(file), temp, path:path.to_owned() }))))
}
pub fn write(path: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> Result<()> {
    let mut writer = open_writer(path.as_ref())?; writer.write_all(bytes.as_ref())?; writer.commit()
}
