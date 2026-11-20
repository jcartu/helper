//! Native Hyprland identity: terminal emulators need not expose AT-SPI.
//!
//! Hyprland has two sockets under `$XDG_RUNTIME_DIR/hypr/<signature>/`:
//!   * `.socket.sock`  — request/response (`j/activewindow`, `j/clients`).
//!   * `.socket2.sock` — a push stream of compositor events, one per line
//!     (`activewindow>>class,title`, `windowtitle>>addr`, ...).
//!
//! Identity is always read from the request socket (the event stream does not
//! carry enough to build an [`ActiveApp`], and its fields are comma-joined so a
//! title containing a comma is ambiguous). The event stream is used only as a
//! wake-up: a focus/title event triggers one query. If the event socket is
//! unavailable the worker falls back to polling.
use super::{ActiveApp, ActiveAppProvider, EventSink, Result, RunningApp};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

/// Longest reply we accept from the request socket.
const REPLY_LIMIT: u64 = 262_144;
/// Bound on every request-socket read/write: a stalled compositor must not
/// stall a paste.
const REQUEST_TIMEOUT: Duration = Duration::from_millis(200);
/// How often the worker wakes to check `stop` while the event stream is idle.
const EVENT_IDLE_TICK: Duration = Duration::from_millis(500);
/// Poll interval when the event socket cannot be used.
const FALLBACK_POLL: Duration = Duration::from_millis(200);
/// An unterminated event line longer than this is discarded (titles are
/// user-controlled; never let one grow the buffer without bound).
const EVENT_LINE_LIMIT: usize = 16_384;

fn request(socket: &Path, command: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(REQUEST_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(REQUEST_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(command.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut text = String::new();
    stream
        .take(REPLY_LIMIT + 1)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    if text.len() as u64 > REPLY_LIMIT {
        return Err("Hyprland reply exceeds limit".into());
    }
    Ok(text)
}

fn parse_window(value: &serde_json::Value) -> Option<ActiveApp> {
    let class = ["class", "initialClass"]
        .into_iter()
        .filter_map(|key| value.get(key)?.as_str())
        .find(|name| !name.trim().is_empty())?;
    Some(ActiveApp {
        app_name: class.to_owned(),
        bundle_id: class.to_owned(),
        window_title: value
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned(),
        url: String::new(),
    })
}

fn parse_active(text: &str) -> Option<ActiveApp> {
    parse_window(&serde_json::from_str::<serde_json::Value>(text).ok()?)
}

/// True for the event-stream lines after which the focused window's identity
/// or title may differ. Everything else on the stream (workspace layout, layer
/// surfaces, monitor hotplug, ...) is ignored without a query.
fn is_focus_event(line: &[u8]) -> bool {
    const PREFIXES: &[&[u8]] = &[
        b"activewindow>>",
        b"activewindowv2>>",
        b"windowtitle>>",
        b"windowtitlev2>>",
        b"closewindow>>",
    ];
    PREFIXES.iter().any(|p| line.starts_with(p))
}

/// Split complete lines off the front of `buf`, returning whether any of them
/// was a focus event. A trailing partial line stays in `buf` for the next read.
fn drain_focus_events(buf: &mut Vec<u8>) -> bool {
    let mut relevant = false;
    while let Some(end) = buf.iter().position(|&b| b == b'\n') {
        relevant |= is_focus_event(&buf[..end]);
        buf.drain(..=end);
    }
    if buf.len() > EVENT_LINE_LIMIT {
        buf.clear();
    }
    relevant
}

type Identity = (String, String, String);

/// State shared by the tracker handle and its worker thread.
struct Shared {
    socket: PathBuf,
    events_socket: PathBuf,
    enabled: AtomicBool,
    stop: AtomicBool,
    /// Identity last announced; `None` forces the next announce through.
    last: Mutex<Option<Identity>>,
    sequence: AtomicU64,
    events: Mutex<EventSink>,
}

impl Shared {
    /// Query the focused window and emit `AppInfoUpdate` if it differs from the
    /// last one announced. No focused window announces an empty identity; a
    /// failed query announces nothing (never a stale or invented app).
    fn announce(&self) {
        let Ok(raw) = request(&self.socket, "j/activewindow") else {
            return;
        };
        let app = parse_active(&raw).unwrap_or_default();
        let key = (
            app.app_name.clone(),
            app.bundle_id.clone(),
            app.window_title.clone(),
        );
        let Ok(mut last) = self.last.lock() else {
            return;
        };
        if last.as_ref() == Some(&key) {
            return;
        }
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let id = format!("hypr-{}-{sequence}", std::process::id());
        let event = crate::proto::request("AppInfoUpdate", super::active_app_payload(&app), &id);
        if let Ok(sink) = self.events.lock() {
            let _ = sink.send(event);
        }
        *last = Some(key);
    }

    fn announce_if_enabled(&self) {
        if self.enabled.load(Ordering::Relaxed) {
            self.announce();
        }
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Follow the event stream until it ends or `stop` is set. Returns `Err`
    /// only when the socket could not be opened at all.
    fn follow_events(&self) -> std::io::Result<()> {
        let mut stream = UnixStream::connect(&self.events_socket)?;
        stream.set_read_timeout(Some(EVENT_IDLE_TICK))?;
        // Anything that changed while we were not listening.
        self.announce_if_enabled();
        let mut pending = Vec::new();
        let mut chunk = [0u8; 4096];
        while !self.stopping() {
            match stream.read(&mut chunk) {
                Ok(0) => return Ok(()), // compositor closed the stream
                Ok(n) => {
                    pending.extend_from_slice(&chunk[..n]);
                    if drain_focus_events(&mut pending) {
                        self.announce_if_enabled();
                    }
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                    ) => {}
                Err(_) => return Ok(()),
            }
        }
        Ok(())
    }

    fn run(&self) {
        let mut warned = false;
        while !self.stopping() {
            if let Err(e) = self.follow_events() {
                if !warned {
                    log::warn!(
                        "Hyprland event socket unavailable ({e}); polling for focus changes"
                    );
                    warned = true;
                }
                self.announce_if_enabled();
            }
            if !self.stopping() {
                std::thread::sleep(FALLBACK_POLL);
            }
        }
    }
}

pub struct HyprlandTracker {
    shared: Arc<Shared>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl HyprlandTracker {
    pub fn start(events: EventSink) -> Result<Self> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").ok_or("Missing XDG_RUNTIME_DIR")?;
        let signature = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").map_err(|e| e.to_string())?;
        if signature.is_empty() || signature.contains('/') || signature == ".." {
            return Err("Invalid Hyprland instance signature".into());
        }
        let dir = PathBuf::from(runtime).join("hypr").join(signature);
        let socket = dir.join(".socket.sock");
        // Empty activewindow ({}) is valid when no surface currently has focus.
        let initial = request(&socket, "j/activewindow")?;
        serde_json::from_str::<serde_json::Value>(&initial).map_err(|e| e.to_string())?;
        let shared = Arc::new(Shared {
            socket,
            events_socket: dir.join(".socket2.sock"),
            enabled: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            last: Mutex::new(None),
            sequence: AtomicU64::new(0),
            events: Mutex::new(events),
        });
        let background = shared.clone();
        let worker = std::thread::Builder::new()
            .name("hyprland-focus".to_string())
            .spawn(move || background.run())
            .map_err(|e| e.to_string())?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }
}

impl ActiveAppProvider for HyprlandTracker {
    fn current(&self) -> Option<ActiveApp> {
        // Always query at paste time: never classify a terminal using stale AT-SPI focus.
        request(&self.shared.socket, "j/activewindow")
            .ok()
            .and_then(|s| parse_active(&s))
    }
    fn get_running_apps(&self) -> Vec<RunningApp> {
        let raw = match request(&self.shared.socket, "j/clients") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let values: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap_or_default();
        let mut seen = std::collections::HashSet::new();
        values
            .iter()
            .filter_map(parse_window)
            .filter(|app| seen.insert(app.bundle_id.clone()))
            .map(|app| RunningApp {
                bundle_id: app.bundle_id,
                name: app.app_name,
            })
            .collect()
    }
    fn set_focus_detection(&self, active: bool) {
        self.shared.enabled.store(active, Ordering::Relaxed);
        // Forget what was announced. Enabling must re-announce even an
        // unchanged app (see RecordingStarted in main.rs): the app clears the
        // dictation's target at every start and refills it only from an
        // AppInfoUpdate that arrives while the dictation is running.
        if let Ok(mut last) = self.shared.last.lock() {
            *last = None;
        }
        if active {
            self.shared.announce();
        }
    }
    fn name(&self) -> &'static str {
        "hyprland-ipc"
    }
}

impl Drop for HyprlandTracker {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;

    fn socket_path(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "wispr-hypr-{tag}-{}-{nonce}.sock",
            std::process::id()
        ))
    }

    fn shared(socket: PathBuf, events_socket: PathBuf, sink: EventSink) -> Arc<Shared> {
        Arc::new(Shared {
            socket,
            events_socket,
            enabled: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            last: Mutex::new(None),
            sequence: AtomicU64::new(0),
            events: Mutex::new(sink),
        })
    }

    /// A request socket that answers every `j/activewindow` with the class
    /// currently stored in `class` (empty string = no focused window).
    fn serve_active(path: &Path, class: Arc<Mutex<String>>) {
        let listener = UnixListener::bind(path).unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut cmd = [0u8; 64];
                let Ok(n) = stream.read(&mut cmd) else {
                    continue;
                };
                assert_eq!(&cmd[..n], b"j/activewindow");
                let name = class.lock().unwrap().clone();
                let body = if name.is_empty() {
                    "{}".to_string()
                } else {
                    format!(r#"{{"class":"{name}","title":"t"}}"#)
                };
                let _ = stream.write_all(body.as_bytes());
            }
        });
    }

    fn announced_app(rx: &mpsc::Receiver<serde_json::Value>) -> String {
        let event = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("AppInfoUpdate");
        event["HelperAPIRequest"]["AppInfoUpdate"]["payload"]["appName"]
            .as_str()
            .expect("appName")
            .to_owned()
    }

    #[test]
    fn native_ipc_queries_current_window_not_cached_focus() {
        let path = socket_path("req");
        let class = Arc::new(Mutex::new("Alacritty".to_string()));
        serve_active(&path, class.clone());
        let (tx, _rx) = mpsc::channel();
        let tracker = HyprlandTracker {
            shared: shared(path.clone(), socket_path("none"), tx),
            worker: None,
        };
        let first = tracker.current().unwrap();
        assert!(crate::paste::looks_like_terminal(
            &first.app_name,
            &first.bundle_id
        ));
        *class.lock().unwrap() = "firefox".to_string();
        let second = tracker.current().unwrap();
        assert!(!crate::paste::looks_like_terminal(
            &second.app_name,
            &second.bundle_id
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn stalled_compositor_is_bounded() {
        let path = socket_path("stall");
        let listener = UnixListener::bind(&path).unwrap();
        let worker = std::thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(500));
        });
        let start = std::time::Instant::now();
        assert!(request(&path, "j/activewindow").is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn omp_title_still_resolves_to_alacritty() {
        let app = parse_active(r#"{"class":"Alacritty","initialClass":"Alacritty","title":"OMP"}"#)
            .unwrap();
        assert_eq!(app.bundle_id, "Alacritty");
        assert_eq!(app.window_title, "OMP");
        assert!(crate::paste::looks_like_terminal(
            &app.app_name,
            &app.bundle_id
        ));
    }

    #[test]
    fn browser_title_cannot_make_it_a_terminal() {
        let app = parse_active(
            r#"{"class":"com.nousresearch.hermes","title":"Alacritty terminal troubleshooting"}"#,
        )
        .unwrap();
        assert!(!crate::paste::looks_like_terminal(
            &app.app_name,
            &app.bundle_id
        ));
    }

    #[test]
    fn initial_class_is_a_fallback() {
        assert_eq!(
            parse_active(r#"{"class":"","initialClass":"kitty","title":"OMP"}"#)
                .unwrap()
                .app_name,
            "kitty"
        );
    }

    #[test]
    fn no_window_or_invalid_data_never_reuses_previous_identity() {
        for raw in ["{}", "null", "[]", "broken", r#"{"title":"OMP"}"#] {
            assert!(parse_active(raw).is_none());
        }
    }

    #[test]
    fn only_focus_and_title_events_trigger_a_query() {
        for line in [
            &b"activewindow>>Alacritty,OMP"[..],
            b"activewindowv2>>55caa329df60",
            b"windowtitle>>55caa329df60",
            b"windowtitlev2>>55caa329df60,a, b",
            b"closewindow>>55caa329df60",
        ] {
            assert!(is_focus_event(line), "{}", String::from_utf8_lossy(line));
        }
        for line in [
            &b"workspace>>3"[..],
            b"openlayer>>waybar",
            b"monitoradded>>DP-1",
            b"fullscreen>>1",
            b"activelayout>>kbd,English (US)",
            b"",
            // A title that merely mentions an event name is not an event.
            b"openwindow>>55caa,1,foot,activewindow>>x",
        ] {
            assert!(!is_focus_event(line), "{}", String::from_utf8_lossy(line));
        }
    }

    #[test]
    fn event_lines_split_across_reads_are_reassembled() {
        let mut buf = b"workspace>>2\nactivewin".to_vec();
        assert!(!drain_focus_events(&mut buf));
        assert_eq!(buf, b"activewin");
        buf.extend_from_slice(b"dow>>kitty,zsh\nopenlayer>>");
        assert!(drain_focus_events(&mut buf));
        assert_eq!(buf, b"openlayer>>");
    }

    #[test]
    fn oversized_partial_line_is_dropped() {
        let mut buf = vec![b'x'; EVENT_LINE_LIMIT + 1];
        assert!(!drain_focus_events(&mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn enabling_reannounces_an_unchanged_app() {
        let path = socket_path("re");
        serve_active(&path, Arc::new(Mutex::new("Alacritty".to_string())));
        let (tx, rx) = mpsc::channel();
        let tracker = HyprlandTracker {
            shared: shared(path.clone(), socket_path("none"), tx),
            worker: None,
        };
        tracker.set_focus_detection(true);
        assert_eq!(announced_app(&rx), "Alacritty");
        // Same window, no change: a plain announce is deduplicated...
        tracker.shared.announce();
        assert!(rx.try_recv().is_err());
        // ...but RecordingStarted re-enables, and that must announce again.
        tracker.set_focus_detection(true);
        assert_eq!(announced_app(&rx), "Alacritty");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disabled_tracker_announces_nothing() {
        let path = socket_path("off");
        serve_active(&path, Arc::new(Mutex::new("Alacritty".to_string())));
        let (tx, rx) = mpsc::channel();
        let state = shared(path.clone(), socket_path("none"), tx);
        state.announce_if_enabled();
        assert!(rx.try_recv().is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn focus_event_on_the_stream_announces_the_new_window() {
        let req = socket_path("evreq");
        let ev = socket_path("evstream");
        let class = Arc::new(Mutex::new("Alacritty".to_string()));
        serve_active(&req, class.clone());
        let stream_listener = UnixListener::bind(&ev).unwrap();
        let (tx, rx) = mpsc::channel();
        let state = shared(req.clone(), ev.clone(), tx);
        state.enabled.store(true, Ordering::Relaxed);
        let background = state.clone();
        let worker = std::thread::spawn(move || background.run());

        let (mut compositor, _) = stream_listener.accept().unwrap();
        // Connecting announces whatever is focused right now.
        assert_eq!(announced_app(&rx), "Alacritty");
        // An unrelated event must not produce an announce.
        compositor.write_all(b"workspace>>2\n").unwrap();
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err());
        // Focus moves: the event wakes the worker, the request socket says where.
        *class.lock().unwrap() = "firefox".to_string();
        compositor.write_all(b"activewindow>>firefox,t\n").unwrap();
        assert_eq!(announced_app(&rx), "firefox");
        // Focus lost entirely: announced as an empty identity, not the old app.
        *class.lock().unwrap() = String::new();
        compositor.write_all(b"closewindow>>55caa\n").unwrap();
        assert_eq!(announced_app(&rx), "");

        state.stop.store(true, Ordering::Relaxed);
        drop(compositor);
        worker.join().unwrap();
        std::fs::remove_file(req).unwrap();
        std::fs::remove_file(ev).unwrap();
    }

    #[test]
    fn missing_event_socket_falls_back_to_polling() {
        let req = socket_path("pollreq");
        let class = Arc::new(Mutex::new("kitty".to_string()));
        serve_active(&req, class.clone());
        let (tx, rx) = mpsc::channel();
        let state = shared(req.clone(), socket_path("absent"), tx);
        state.enabled.store(true, Ordering::Relaxed);
        let background = state.clone();
        let worker = std::thread::spawn(move || background.run());
        assert_eq!(announced_app(&rx), "kitty");
        *class.lock().unwrap() = "firefox".to_string();
        assert_eq!(announced_app(&rx), "firefox");
        state.stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();
        std::fs::remove_file(req).unwrap();
    }
}
