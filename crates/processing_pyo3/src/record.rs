//! `begin_record` / `end_record`: capture drawn frames into a video file.

use std::cell::RefCell;

use bevy::prelude::Entity;
use processing_video::{VideoRecorder, VideoRecorderConfig};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const X264_PRESETS: [&str; 10] = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
];

enum RecState {
    /// The recorder is created on the first captured frame so its dimensions
    /// match the actual readback.
    Pending {
        path: String,
        fps: f64,
        crf: Option<u8>,
        preset: Option<String>,
    },
    Active(VideoRecorder),
}

thread_local! {
    static RECORDER: RefCell<Option<RecState>> = const { RefCell::new(None) };
}

pub(crate) fn begin(path: &str, fps: f64, crf: Option<u8>, preset: Option<String>) -> PyResult<()> {
    RECORDER.with(|r| {
        let mut slot = r.borrow_mut();
        if slot.is_some() {
            return Err(PyRuntimeError::new_err(
                "already recording; call end_record() first",
            ));
        }
        if !(fps.is_finite() && fps > 0.0 && fps <= 1_000_000.0) {
            return Err(PyValueError::new_err(format!(
                "fps must be a positive number no greater than 1000000, got {fps}"
            )));
        }
        if let Some(crf) = crf
            && crf > 51
        {
            return Err(PyValueError::new_err(format!(
                "crf must be between 0 (lossless) and 51 (worst quality), got {crf}"
            )));
        }
        if let Some(ref preset) = preset
            && !X264_PRESETS.contains(&preset.as_str())
        {
            return Err(PyValueError::new_err(format!(
                "unknown preset {preset:?}; expected one of {X264_PRESETS:?}"
            )));
        }
        *slot = Some(RecState::Pending {
            path: path.to_string(),
            fps,
            crf,
            preset,
        });
        Ok(())
    })
}

pub(crate) fn capture(entity: Entity) -> PyResult<()> {
    RECORDER.with(|r| {
        let mut slot = r.borrow_mut();
        let Some(state) = slot.as_mut() else {
            return Ok(());
        };
        let result = (|| {
            let (rgba, width, height) = crate::graphics::readback_rgba8(entity)?;
            if let RecState::Pending {
                path,
                fps,
                crf,
                preset,
            } = state
            {
                let mut config = VideoRecorderConfig::new(width, height, *fps);
                config.crf = *crf;
                config.preset = preset.clone();
                let recorder = VideoRecorder::new(&*path, config)
                    .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                *state = RecState::Active(recorder);
            }
            let RecState::Active(recorder) = state else {
                unreachable!("state was just made Active");
            };
            recorder
                .record_frame(rgba)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
        })();
        if result.is_err() {
            *slot = None;
        }
        result
    })
}

/// Finish the recording and return the number of frames encoded.
pub(crate) fn end() -> PyResult<u64> {
    RECORDER.with(|r| match r.borrow_mut().take() {
        None => Err(PyRuntimeError::new_err(
            "not recording; call begin_record() first",
        )),
        Some(RecState::Pending { .. }) => Err(PyRuntimeError::new_err(
            "no frames were drawn between begin_record() and end_record()",
        )),
        Some(RecState::Active(recorder)) => recorder
            .finish()
            .map_err(|e| PyRuntimeError::new_err(format!("{e}"))),
    })
}

/// Closing the window mid-recording still leaves a playable file.
pub(crate) fn finish_on_exit() {
    RECORDER.with(|r| {
        if let Some(RecState::Active(recorder)) = r.borrow_mut().take() {
            let _ = recorder.finish();
        }
    });
}
