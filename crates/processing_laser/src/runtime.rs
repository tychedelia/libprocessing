//! The engine thread: owns preset state (staged/live), four engine instances,
//! the audio analysis bus, and the audio-interface DAC. FFI callers talk to it
//! through `Cmd` messages; continuous outputs (frames, preset JSON, audio
//! bands, status) are published as lock-free ArcSwap snapshots, so reads never
//! block the realtime path.
//!
//! Routing (the REPL preview/program model): the STAGED pair renders
//! `state.preview` for the host's preview surface; the LIVE pair renders
//! `state.live` and feeds the DAC. `commit` promotes staged to live.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use laser_engine::engine::{BusSnapshot, Engine, Frame};
use laser_engine::model::{DivergenceConfig, Preset, resolve, save_bank};
use laser_rt::app::AppState;
use laser_rt::audio::AudioBus;
use laser_rt::dac::AudioDac;
use laser_rt::settings::Settings;

const TICK_HZ: f64 = 60.0;

pub enum Cmd {
    StagePreset(Box<Preset>),
    Commit,
    Revert,
    Recall(usize),
    Store(usize),
    SaveBank,
    SetMaster(f64),
    SetEnable(bool, bool),
    SetMirror(bool),
    AudioStart(Option<String>),
    AudioStop,
    /// device name (None = settings.laser_output), rate (0 = settings.output_rate)
    DacStart(Option<String>, u32),
    DacStop,
    Shutdown,
}

/// Frame slot indices for `Shared::frames` / `laser_frame_read`.
pub const FRAME_STAGED_A: usize = 0;
pub const FRAME_STAGED_B: usize = 1;
pub const FRAME_LIVE_A: usize = 2;
pub const FRAME_LIVE_B: usize = 3;

pub struct Shared {
    /// staged A, staged B, live A, live B
    pub frames: [ArcSwap<Frame>; 4],
    pub staged_json: ArcSwap<String>,
    pub live_json: ArcSwap<String>,
    pub bank_json: ArcSwap<String>,
    pub dac_status: ArcSwap<String>,
    /// errors from asynchronous commands (audio/DAC start), newest wins
    pub async_error: ArcSwap<String>,
    /// smoothed bands [0..5] + raw bands [5..10]
    pub audio_bands: ArcSwap<[f32; 10]>,
    pub audio_running: AtomicBool,
    /// staged differs from live (uncommitted edits)
    pub dirty: AtomicBool,
}

impl Shared {
    fn new() -> Shared {
        Shared {
            frames: [
                ArcSwap::from_pointee(Frame::default()),
                ArcSwap::from_pointee(Frame::default()),
                ArcSwap::from_pointee(Frame::default()),
                ArcSwap::from_pointee(Frame::default()),
            ],
            staged_json: ArcSwap::from_pointee(String::new()),
            live_json: ArcSwap::from_pointee(String::new()),
            bank_json: ArcSwap::from_pointee(String::new()),
            dac_status: ArcSwap::from_pointee(String::new()),
            async_error: ArcSwap::from_pointee(String::new()),
            audio_bands: ArcSwap::from_pointee([0.0; 10]),
            audio_running: AtomicBool::new(false),
            dirty: AtomicBool::new(false),
        }
    }
}

pub struct Runtime {
    pub tx: Sender<Cmd>,
    pub shared: Arc<Shared>,
    pub join: Option<JoinHandle<()>>,
}

pub struct Options {
    pub bank_path: Option<PathBuf>,
    pub settings_path: Option<PathBuf>,
}

pub fn spawn(opts: Options) -> Result<Runtime, String> {
    let (tx, rx) = channel();
    let shared = Arc::new(Shared::new());
    let sh = shared.clone();
    let join = std::thread::Builder::new()
        .name("laser-engine".into())
        .spawn(move || run(rx, sh, opts))
        .map_err(|e| format!("failed to spawn laser engine thread: {e}"))?;
    Ok(Runtime { tx, shared, join: Some(join) })
}

fn load_settings(path: Option<&Path>) -> Settings {
    let path = path.unwrap_or_else(|| Path::new("settings.json"));
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Engine B's effective LIVE preset: the committed preset resolved through its
/// own stored divergence at the current master (the live twin of
/// `AppState::resolve_b`, which resolves the staged side).
fn resolve_live_b(state: &AppState) -> Preset {
    let div = DivergenceConfig::from_value(state.live.divergence.as_ref())
        .to_divergence(state.live.mods.len());
    resolve(&state.live, &div, Some(state.master))
}

fn mirrored(mut f: Frame) -> Frame {
    for s in &mut f.strips {
        for p in s {
            p.pos[0] = -p.pos[0];
        }
    }
    f
}

fn bus_snapshot(audio: &Option<AudioBus>, n_mods: usize) -> BusSnapshot {
    let mut bus = BusSnapshot::default();
    if let Some(a) = audio {
        let f = a.snapshot();
        bus.bands = f.smooth;
        bus.bands_raw = f.raw;
        bus.spectrum = f.spectrum.clone();
    }
    bus.fire = vec![false; n_mods];
    bus
}

fn publish_state(shared: &Shared, state: &AppState) {
    let mut staged = state.preview.clone();
    staged.divergence = state.div.to_value();
    shared
        .staged_json
        .store(Arc::new(serde_json::to_string(&staged).unwrap_or_default()));
    shared
        .live_json
        .store(Arc::new(serde_json::to_string(&state.live).unwrap_or_default()));
    shared
        .bank_json
        .store(Arc::new(serde_json::to_string(&state.bank).unwrap_or_default()));
    shared.dirty.store(state.dirty(), Ordering::Relaxed);
}

fn async_error(shared: &Shared, msg: String) {
    shared.async_error.store(Arc::new(msg));
}

fn run(rx: Receiver<Cmd>, shared: Arc<Shared>, opts: Options) {
    let bank_path = opts.bank_path.unwrap_or_else(|| PathBuf::from("presets.json"));
    let mut settings = load_settings(opts.settings_path.as_deref());

    let mut state = AppState::load_from(bank_path.clone(), bank_path);
    state.mirror = settings.mirror;
    state.enable_a = settings.enable_a;
    state.enable_b = settings.enable_b;
    state.master = settings.master;

    let mut staged = (Engine::new(0xA), Engine::new(0xB));
    let mut live = (Engine::new(0xA), Engine::new(0xB));
    let mut audio: Option<AudioBus> = None;
    let mut dac: Option<AudioDac> = None;

    publish_state(&shared, &state);

    let epoch = Instant::now();
    let tick = Duration::from_secs_f64(1.0 / TICK_HZ);
    let mut next = Instant::now() + tick;
    loop {
        let mut dirty = false;
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                Cmd::Shutdown => return,
                Cmd::StagePreset(p) => {
                    state.div = DivergenceConfig::from_value(p.divergence.as_ref());
                    state.preview = *p;
                    dirty = true;
                }
                Cmd::Commit => {
                    state.commit();
                    dirty = true;
                }
                Cmd::Revert => {
                    state.revert();
                    dirty = true;
                }
                Cmd::Recall(slot) => {
                    state.recall(slot);
                    dirty = true;
                }
                Cmd::Store(slot) => {
                    state.store(slot);
                    dirty = true;
                }
                Cmd::SaveBank => {
                    if let Err(e) = save_bank(&state.bank_path, &state.bank) {
                        async_error(&shared, format!("save_bank failed: {e}"));
                    }
                }
                Cmd::SetMaster(v) => {
                    state.master = v;
                    dirty = true;
                }
                Cmd::SetEnable(a, b) => {
                    state.enable_a = a;
                    state.enable_b = b;
                }
                Cmd::SetMirror(m) => state.mirror = m,
                Cmd::AudioStart(pref) => {
                    let pref = pref.or_else(|| settings.audio_input.clone());
                    match AudioBus::start_with(pref.as_deref()) {
                        Ok(bus) => {
                            audio = Some(bus);
                            shared.audio_running.store(true, Ordering::Relaxed);
                        }
                        Err(e) => {
                            audio = None;
                            shared.audio_running.store(false, Ordering::Relaxed);
                            async_error(&shared, format!("audio start failed: {e}"));
                        }
                    }
                }
                Cmd::AudioStop => {
                    audio = None;
                    shared.audio_running.store(false, Ordering::Relaxed);
                }
                Cmd::DacStart(name, rate) => {
                    if rate != 0 {
                        settings.output_rate = rate;
                    }
                    match name.or_else(|| settings.laser_output.clone()) {
                        Some(n) => {
                            settings.laser_output = Some(n.clone());
                            let d = AudioDac::start(&n, settings.output_rate);
                            shared.dac_status.store(Arc::new(d.status.clone()));
                            dac = Some(d);
                        }
                        None => async_error(
                            &shared,
                            "no DAC device: pass a name or set laser_output in settings".into(),
                        ),
                    }
                }
                Cmd::DacStop => {
                    dac = None;
                    shared.dac_status.store(Arc::new(String::new()));
                }
            }
        }
        if dirty {
            publish_state(&shared, &state);
        }

        let t = epoch.elapsed().as_secs_f64();
        let bus = bus_snapshot(&audio, state.preview.mods.len());
        if let Some(a) = &audio {
            let f = a.snapshot();
            let mut bands = [0.0f32; 10];
            bands[..5].copy_from_slice(&f.smooth);
            bands[5..].copy_from_slice(&f.raw);
            shared.audio_bands.store(Arc::new(bands));
        }

        // staged pair -> preview snapshots
        let p = &state.preview;
        let sa = if state.enable_a {
            staged.0.eval(&p.engine, &p.mods, p.ramp.as_deref(), t, &bus)
        } else {
            Frame::default()
        };
        let sb = if state.enable_b {
            let pb = state.resolve_b();
            staged.1.eval(&pb.engine, &pb.mods, pb.ramp.as_deref(), t, &bus)
        } else {
            Frame::default()
        };

        // live pair -> the DAC
        let l = &state.live;
        let la = if state.enable_a {
            live.0.eval(&l.engine, &l.mods, l.ramp.as_deref(), t, &bus)
        } else {
            Frame::default()
        };
        let lb = if state.enable_b {
            let pb = resolve_live_b(&state);
            live.1.eval(&pb.engine, &pb.mods, pb.ramp.as_deref(), t, &bus)
        } else {
            Frame::default()
        };

        if let Some(d) = &dac {
            if d.active() {
                let dac_b = if state.mirror { mirrored(lb.clone()) } else { lb.clone() };
                d.push(
                    &la,
                    &dac_b,
                    [state.enable_a, state.enable_b],
                    settings.output_rate,
                    [&settings.projectors[0], &settings.projectors[1]],
                    &settings.tuning,
                );
            }
            if **shared.dac_status.load() != d.status {
                shared.dac_status.store(Arc::new(d.status.clone()));
            }
        }

        shared.frames[FRAME_STAGED_A].store(Arc::new(sa));
        shared.frames[FRAME_STAGED_B].store(Arc::new(sb));
        shared.frames[FRAME_LIVE_A].store(Arc::new(la));
        shared.frames[FRAME_LIVE_B].store(Arc::new(lb));

        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        }
        next += tick;
        if next < Instant::now() {
            // fell behind (debugger, contention): re-anchor instead of spiraling
            next = Instant::now() + tick;
        }
    }
}
