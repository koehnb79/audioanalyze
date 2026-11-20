#!/usr/bin/env python3
"""Guitar Rhythm Analyzer 0.9.4.1-dev: local-grid and metronome-reference diagnostics."""
from __future__ import annotations
import argparse, csv, json, math, re, shutil, sqlite3, subprocess, sys, tempfile, traceback
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
from typing import Optional
import librosa
import numpy as np
from local_grid_analysis import analyze_local_grid
from metronome_reference_analysis import detect_metronome_reference

AUDIO_EXTENSIONS = {'.wav','.mp3','.flac','.m4a','.aac','.ogg','.aiff','.aif'}
VIDEO_EXTENSIONS = {'.mp4','.mov','.mkv','.avi','.webm','.mpeg','.mpg','.m4v'}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
ANALYSIS_VERSION = '0.9.4.2-dev'


def extract_video_audio(video_path: Path, wav_path: Path) -> None:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("FFmpeg was not found; install it and make sure 'ffmpeg' is on PATH.")
    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video_path),'-vn','-ac','1','-ar','22050','-c:a','pcm_s16le',str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'FFmpeg could not extract audio.')


def load_recording(path: Path) -> tuple[np.ndarray,int]:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        with tempfile.TemporaryDirectory(prefix='guitar_rhythm_') as folder:
            wav = Path(folder)/'audio.wav'
            extract_video_audio(path,wav)
            y,sr = librosa.load(wav,sr=22050,mono=True)
    else:
        try:
            y,sr = librosa.load(path,sr=22050,mono=True)
        except Exception as first_error:
            if shutil.which('ffmpeg') is None:
                raise first_error
            with tempfile.TemporaryDirectory(prefix='guitar_rhythm_') as folder:
                wav = Path(folder)/'converted.wav'
                cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(path),'-vn','-ac','1','-ar','22050','-c:a','pcm_s16le',str(wav)]
                result = subprocess.run(cmd,capture_output=True,text=True)
                if result.returncode:
                    raise RuntimeError(f'Audio decoder and FFmpeg both failed: {first_error}; {result.stderr.strip()}')
                y,sr = librosa.load(wav,sr=22050,mono=True)
    if not y.size: raise RuntimeError('No audio samples could be decoded.')
    return y,sr


def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None


def cv(values: np.ndarray) -> Optional[float]:
    if values.size < 3: return None
    mean=float(np.mean(values))
    return finite(np.std(values,ddof=1)/mean*100) if mean>0 else None


def phrase_groups(onsets: np.ndarray) -> tuple[list[np.ndarray],float]:
    """Split onsets at unusually long gaps; threshold is data-adaptive and capped."""
    if onsets.size < 2: return ([onsets] if onsets.size else []), 0.8
    gaps=np.diff(onsets)
    core=gaps[gaps <= np.quantile(gaps,0.75)]
    reference=float(np.median(core)) if core.size else float(np.median(gaps))
    threshold=float(np.clip(max(0.8,reference*3.0),0.8,2.5))
    cuts=np.where(gaps>threshold)[0]+1
    return [g for g in np.split(onsets,cuts) if g.size],threshold


def recording_date(path: Path) -> tuple[Optional[str], str]:
    """Prefer embedded creation time, then filename date, then filesystem mtime."""
    if shutil.which('ffprobe'):
        try:
            cmd=['ffprobe','-v','error','-show_entries','format_tags=creation_time:stream_tags=creation_time','-of','json',str(path)]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            if result.returncode == 0:
                data=json.loads(result.stdout or '{}')
                tags=dict(data.get('format',{}).get('tags',{}) or {})
                for stream in data.get('streams',[]): tags.update(stream.get('tags',{}) or {})
                raw=tags.get('creation_time')
                if raw:
                    dt=datetime.fromisoformat(raw.replace('Z','+00:00'))
                    if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone().isoformat(timespec='seconds'),'embedded_creation_time'
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError): pass
    for pattern in (r'(?<!\d)(20\d{2})[-_](\d{1,2})[-_](\d{1,2})(?!\d)',r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)'):
        m=re.search(pattern,path.stem)
        if m:
            try:
                dt=datetime(int(m.group(1)),int(m.group(2)),int(m.group(3))).astimezone()
                return dt.isoformat(timespec='seconds'),'filename_date'
            except ValueError: pass
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec='seconds'),'file_modified_fallback'


def tempo_windows(y: np.ndarray, sr: int, window_seconds: float=30.0, hop_seconds: float=15.0) -> list[dict]:
    """Estimate local tempo in overlapping windows; not a ground-truth beat tracker."""
    win=int(window_seconds*sr); step=int(hop_seconds*sr); hop=256
    if len(y)<win: return []
    out=[]
    for start in range(0,len(y)-win+1,step):
        env=librosa.onset.onset_strength(y=y[start:start+win],sr=sr,hop_length=hop,aggregate=np.median)
        tempo,beats=librosa.beat.beat_track(onset_envelope=env,sr=sr,hop_length=hop,units='time')
        bpm=float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else None
        local_candidates=pulse_autocorrelation_candidates(env,sr,hop)
        if bpm and math.isfinite(bpm) and bpm>0:
            out.append({'start_seconds':round(start/sr,2),'end_seconds':round((start+win)/sr,2),
                        'tempo_bpm':round(bpm,2),'beat_count':int(len(beats)),
                        'autocorrelation_candidates':local_candidates})
    return out


def plot_diagnostics(y: np.ndarray, sr: int, onsets: np.ndarray, phrases: list[np.ndarray], out: Path, stem: str, candidate_clicks: Optional[np.ndarray]=None) -> Optional[str]:
    """Save onset diagnostics and visually distinguish unverified click candidates."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out.mkdir(parents=True,exist_ok=True)
        times=np.arange(y.size)/sr
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,6),sharex=False,constrained_layout=True)
        step=max(1,int(sr/3000))
        ax1.plot(times[::step],y[::step],linewidth=.45)
        for t in onsets: ax1.axvline(t,alpha=.22,linewidth=.55,color='0.45')
        if candidate_clicks is not None and len(candidate_clicks):
            for index,t in enumerate(candidate_clicks):
                ax1.axvline(t,alpha=.9,linewidth=1.0,linestyle='--',color='tab:red',
                            label='Unverified click candidate' if index==0 else None)
            ax1.legend(fontsize=8)
        ax1.set(title='Waveform: detected onsets and unverified click candidates',xlabel='Time (s)',ylabel='Amplitude')
        for i,p in enumerate(phrases):
            if p.size>1: ax2.scatter(p[1:],np.diff(p),s=12,label=f'Phrase {i+1}')
        ax2.set(title='Inter-onset intervals within detected phrases',xlabel='Time of second onset (s)',ylabel='Interval (s)')
        if len(phrases)<=12 and phrases: ax2.legend(fontsize=7,ncol=3)
        target=out/(stem+'_onsets.png')
        fig.savefig(target,dpi=130)
        plt.close(fig)
        return str(target)
    except Exception as exc:
        return f'PLOT_ERROR: {exc}'



def pulse_autocorrelation_candidates(env: np.ndarray, sr: int, hop: int=256,
                                     min_bpm: float=30.0, max_bpm: float=240.0,
                                     limit: int=8) -> list[dict]:
    """Return independent periodicity candidates from onset-envelope autocorrelation.

    These are candidate periodicities, not confirmed musical beats. Peaks at half/double
    tempo are common and must be interpreted alongside listening and beat tracking.
    """
    env=np.asarray(env,dtype=float)
    if env.size < 8 or not np.any(np.isfinite(env)):
        return []
    env=np.nan_to_num(env-np.mean(env))
    ac=np.asarray(librosa.autocorrelate(env,max_size=env.size-1),dtype=float)
    if ac.size < 3 or ac[0] <= 0:
        return []
    ac=ac/ac[0]
    min_lag=max(1,int(round((60.0/max_bpm)*sr/hop)))
    max_lag=min(len(ac)-2,int(round((60.0/min_bpm)*sr/hop)))
    if max_lag <= min_lag:
        return []
    peaks=[]
    for lag in range(min_lag+1,max_lag):
        if ac[lag] > 0.0 and ac[lag] >= ac[lag-1] and ac[lag] > ac[lag+1]:
            peaks.append((float(ac[lag]),lag))
    peaks.sort(reverse=True)
    selected=[]
    for score,lag in peaks:
        bpm=60.0*sr/(lag*hop)
        if any(abs(math.log2(bpm/item['bpm'])) < 0.035 for item in selected):
            continue
        selected.append({'bpm':round(bpm,3),'period_seconds':round(lag*hop/sr,4),
                         'normalized_autocorrelation':round(score,5)})
        if len(selected)>=limit:
            break
    return selected


def octave_equivalent_difference(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Smallest percentage difference after allowing half/double-time equivalence."""
    if not a or not b or a<=0 or b<=0:
        return None
    ratios=[a/b, a/(b*2.0), (a*2.0)/b]
    return round(min(abs(r-1.0) for r in ratios)*100.0,2)




def fitted_grid_metrics(onsets: np.ndarray, beats: np.ndarray) -> dict:
    """Measure beat/onset deviations from a constant grid fitted to detected beats."""
    onsets=np.asarray(onsets,dtype=float); beats=np.asarray(beats,dtype=float)
    result={'fitted_grid_bpm':None,'beat_grid_median_abs_error_ms':None,
      'beat_grid_median_signed_error_ms':None,'beat_grid_p90_abs_error_ms':None,
      'beat_grid_within_50ms_percent':None,'beat_grid_residuals_ms_json':'[]',
      'onset_quarter_grid_median_abs_error_ms':None,
      'onset_quarter_grid_median_signed_error_ms':None,
      'onset_quarter_grid_p90_abs_error_ms':None,
      'onset_quarter_grid_within_50ms_percent':None,
      'onset_quarter_grid_residuals_ms_json':'[]','grid_fit_status':'insufficient_detected_beats'}
    if beats.size<4: return result
    period,phase=np.polyfit(np.arange(beats.size,dtype=float),beats,1)
    if not math.isfinite(period) or period<=0:
        result['grid_fit_status']='invalid_fitted_period'; return result
    beat_res=(beats-(phase+np.arange(beats.size)*period))*1000.0
    result.update({'fitted_grid_bpm':round(60.0/period,3),
      'beat_grid_median_abs_error_ms':round(float(np.median(np.abs(beat_res))),3),
      'beat_grid_median_signed_error_ms':round(float(np.median(beat_res)),3),
      'beat_grid_p90_abs_error_ms':round(float(np.percentile(np.abs(beat_res),90)),3),
      'beat_grid_within_50ms_percent':round(float(np.mean(np.abs(beat_res)<=50)*100),2),
      'beat_grid_residuals_ms_json':json.dumps(np.round(beat_res,3).tolist()),
      'grid_fit_status':'fitted_to_detected_beat_track'})
    if onsets.size:
        sub=period/4.0; phase_mod=phase % sub
        nearest=np.round((onsets-phase_mod)/sub)*sub+phase_mod
        res=(onsets-nearest)*1000.0
        result.update({'onset_quarter_grid_median_abs_error_ms':round(float(np.median(np.abs(res))),3),
          'onset_quarter_grid_median_signed_error_ms':round(float(np.median(res)),3),
          'onset_quarter_grid_p90_abs_error_ms':round(float(np.percentile(np.abs(res),90)),3),
          'onset_quarter_grid_within_50ms_percent':round(float(np.mean(np.abs(res)<=50)*100),2),
          'onset_quarter_grid_residuals_ms_json':json.dumps(np.round(res,3).tolist())})
    return result

def pulse_analysis(y, sr, onsets, mode='self-paced', reference_bpm=None):
    """Estimate internal pulse regularity from the beat track; exploratory, not ground truth."""
    hop=256
    env=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop,aggregate=np.median)
    bpm,beats=librosa.beat.beat_track(onset_envelope=env,sr=sr,hop_length=hop,units='time')
    autocorr_candidates=pulse_autocorrelation_candidates(env,sr,hop)
    bpm=float(np.asarray(bpm).reshape(-1)[0]) if np.asarray(bpm).size else None
    beats=np.asarray(beats,dtype=float)
    if reference_bpm is not None: bpm=float(reference_bpm); source='user_supplied_reference_bpm'
    else: source='self_paced_beat_tracker'

    all_beat_intervals=np.diff(beats)
    valid_interval_mask=(all_beat_intervals>=0.25)&(all_beat_intervals<=2.0)
    beat_intervals=all_beat_intervals[valid_interval_mask]
    beat_cv=cv(beat_intervals) if beat_intervals.size else None
    beat_median=float(np.median(beat_intervals)) if beat_intervals.size else None
    beat_mad_ms=(float(np.median(np.abs(beat_intervals-beat_median))*1000)
                 if beat_intervals.size and beat_median is not None else None)

    # Descriptive drift: slope of local beat intervals over elapsed time.
    # This is not a phase-error measure and can reflect tempo changes or tracker errors.
    drift_ms_per_min=None
    if beat_intervals.size>=5 and len(beats)>=len(beat_intervals)+1:
        interval_times=beats[1:][valid_interval_mask]
        drift_ms_per_min=float(np.polyfit(interval_times/60.0,beat_intervals*1000.0,1)[0])

    duration=max(float(len(y))/sr,1e-9)
    beat_coverage=(float((beats[-1]-beats[0])/duration*100)
                   if len(beats)>=2 else None)
    grid_metrics=fitted_grid_metrics(onsets,beats)
    local_grid_metrics=analyze_local_grid(onsets,beats)
    return {
      **grid_metrics,
      **local_grid_metrics,
      'analysis_mode':mode,'pulse_source':source,'pulse_confidence':'tempo_estimate_only',
      'pulse_bpm':round(bpm,3) if bpm and math.isfinite(bpm) else None,
      'reference_candidate_bpm':None,'self_paced_candidate_bpm':round(bpm,3) if bpm and math.isfinite(bpm) else None,
      'pulse_times_seconds_json':json.dumps(beats.round(4).tolist()),
      'pulse_interval_cv_percent':round(beat_cv,3) if beat_cv is not None else None,
      'pulse_autocorrelation_candidates_json':json.dumps(autocorr_candidates,separators=(',',':')),
      'pulse_autocorrelation_top_bpm':autocorr_candidates[0]['bpm'] if autocorr_candidates else None,
      'tracker_vs_autocorrelation_octave_adjusted_difference_percent':octave_equivalent_difference(bpm,autocorr_candidates[0]['bpm']) if autocorr_candidates else None,
      'beat_interval_count':int(beat_intervals.size),
      'beat_interval_median_ms':round(beat_median*1000,3) if beat_median is not None else None,
      'beat_interval_cv_percent':round(beat_cv,3) if beat_cv is not None else None,
      'beat_interval_mad_ms':round(beat_mad_ms,3) if beat_mad_ms is not None else None,
      'beat_interval_drift_ms_per_min':round(drift_ms_per_min,3) if drift_ms_per_min is not None else None,
      'beat_tracker_time_span_percent':round(beat_coverage,2) if beat_coverage is not None else None,
      'beat_tracker_coverage_percent':round(beat_coverage,2) if beat_coverage is not None else None,
      'pulse_metric_caveat':'Legacy beat-grid residuals are retained for v0.9.2 comparisons. The v0.9.3 local-grid event report provides overlapping-window estimates, explicit beat/subdivision identities, and signed residuals. Both remain anchored to beat tracking from this recording rather than an independent external clock. Autocorrelation candidates are alternative periodicities, not confirmed beats. beat_tracker_coverage_percent is retained for compatibility but actually means time span between first and last detected beats.'}


def load_recording_annotations(folder: Path) -> dict:
    """Load optional human labels from recording_annotations.csv beside the script."""
    annotation_path=Path(__file__).with_name('recording_annotations.csv')
    labels={}
    if annotation_path.exists():
        with annotation_path.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name=(row.get('filename') or '').strip()
                if name:
                    context=(row.get('context') or 'uncertain').strip().lower()
                    if context not in {'solo','accompaniment','uncertain'}:
                        context='uncertain'
                    labels[name]=(context,(row.get('notes') or '').strip())
    return labels


def analyze_recording(path: Path, min_duration: float=3.0, min_onsets: int=5, plot_dir: Optional[Path]=None, mode: str='self-paced', reference_bpm: Optional[float]=None) -> dict:
    y,sr=load_recording(path)
    duration=float(librosa.get_duration(y=y,sr=sr))
    if duration+0.01 < min_duration:
        raise RuntimeError(f'Recording is {duration:.2f}s; minimum is {min_duration:.1f}s.')
    hop=256
    env=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop,aggregate=np.median)
    frames=librosa.onset.onset_detect(onset_envelope=env,sr=sr,hop_length=hop,backtrack=False,units='frames',wait=2)
    onsets=np.asarray(librosa.frames_to_time(frames,sr=sr,hop_length=hop),dtype=float)
    if onsets.size: onsets=onsets[np.r_[True,np.diff(onsets)>=0.045]]
    phrases,gap_threshold=phrase_groups(onsets)
    pulse_metrics=pulse_analysis(y,sr,onsets,mode,reference_bpm)
    # Only attempt independent click detection for recordings explicitly tagged
    # by the user. Repeated drum transients in other recordings are not a click reference.
    if re.search(r'\[Metronome\]', path.stem, flags=re.IGNORECASE):
        pulse_metrics.update(detect_metronome_reference(y, sr, onsets))
    else:
        pulse_metrics.update({
            'metronome_detection_status': 'not_requested_by_recording_tag',
            'metronome_candidate_count': 0,
            'metronome_candidate_times_seconds_json': '[]',
            'metronome_detection_note': 'Independent click detection is attempted only for filenames tagged [Metronome].',
        })
    tempo_profile=tempo_windows(y,sr)
    window_tempos=np.asarray([x['tempo_bpm'] for x in tempo_profile],dtype=float)
    all_intervals=np.diff(onsets)
    within=[]; phrase_stats=[]
    for index,p in enumerate(phrases,1):
        ints=np.diff(p)
        if ints.size:
            within.extend(ints.tolist())
            phrase_stats.append({'phrase':index,'start_seconds':round(float(p[0]),4),'end_seconds':round(float(p[-1]),4),'onsets':int(p.size),'intervals':int(ints.size),'median_interval_seconds':finite(np.median(ints)),'interval_cv_percent':cv(ints)})
    within_arr=np.asarray(within,dtype=float)
    valid_all=all_intervals[(all_intervals>=0.045)&(all_intervals<=3.0)]
    tempo=librosa.feature.tempo(onset_envelope=env,sr=sr,hop_length=hop,aggregate=None)
    tempo=np.asarray(tempo,dtype=float).reshape(-1); tempo=tempo[np.isfinite(tempo)&(tempo>0)]
    phrase_cvs=[p['interval_cv_percent'] for p in phrase_stats if p['interval_cv_percent'] is not None]
    phrase_medians=[p['median_interval_seconds'] for p in phrase_stats if p['median_interval_seconds'] is not None]
    if onsets.size < min_onsets:
        quality='low'; quality_note=f'Fewer than {min_onsets} detected onsets.'
    elif not phrase_stats:
        quality='low'; quality_note='No phrase contains enough detected attacks to calculate intervals.'
    elif len(phrase_stats)==1 and phrase_stats[0]['intervals']<4:
        quality='limited'; quality_note='Only a few within-phrase intervals are available.'
    else:
        quality='usable'; quality_note='Review onset plot; quality label reflects data quantity only.'
    file_stat=path.stat()
    annotation=load_recording_annotations(path.parent).get(path.name,('uncertain',''))
    recording_date_value,recording_date_source=recording_date(path)
    candidate_clicks=np.asarray(json.loads(pulse_metrics.get('metronome_candidate_times_seconds_json','[]')),dtype=float)
    plot_path=plot_diagnostics(y,sr,onsets,phrases,plot_dir,path.stem,candidate_clicks) if plot_dir else None
    return {
      'filename':path.name,'path':str(path.resolve()),'recording_context':annotation[0],'recording_context_notes':annotation[1],'media_type':'video' if path.suffix.lower() in VIDEO_EXTENSIONS else 'audio',
      'file_modified_local':datetime.fromtimestamp(file_stat.st_mtime).astimezone().isoformat(timespec='seconds'),
      'recording_date':recording_date_value,'recording_date_source':recording_date_source,
      'recording_year':int(recording_date_value[:4]) if recording_date_value else None,
      'duration_seconds':round(duration,3),'sample_rate_hz':sr,'detected_onsets':int(onsets.size),'usable_intervals':int(valid_all.size),
      'phrase_count':len(phrase_stats),'phrase_gap_threshold_seconds':round(gap_threshold,3),
      'within_phrase_interval_count':int(within_arr.size),'within_phrase_interval_cv_percent':cv(within_arr),
      'median_phrase_interval_cv_percent':finite(np.median(phrase_cvs)) if phrase_cvs else None,
      'median_phrase_interval_seconds':finite(np.median(phrase_medians)) if phrase_medians else None,
      'within_phrase_interval_sd_ms':finite(np.std(within_arr,ddof=1)*1000) if within_arr.size>1 else None,
      'legacy_inter_onset_cv_percent':cv(valid_all),
      'mean_inter_onset_seconds':finite(np.mean(valid_all)) if valid_all.size else None,
      'median_inter_onset_seconds':finite(np.median(valid_all)) if valid_all.size else None,
      'estimated_tempo_median_bpm':finite(np.median(tempo)) if tempo.size else None,
      'estimated_tempo_sd_bpm':finite(np.std(tempo,ddof=1)) if tempo.size>1 else None,
      **pulse_metrics,
      'tempo_window_seconds':30,'tempo_window_hop_seconds':15,
      'tempo_windows_json':json.dumps(tempo_profile,separators=(',',':')),
      'tempo_window_cv_percent':cv(window_tempos) if window_tempos.size>=3 else None,
      'quality_flag':quality,'quality_note':quality_note,'plot_path':plot_path,
      'phrase_details_json':json.dumps(phrase_stats,separators=(',',':')),
      'onset_times_seconds':json.dumps([round(float(t),4) for t in onsets]),
      'analysis_version':ANALYSIS_VERSION,
      'interpretation_note':'Phrase-aware exploratory estimates; onset detections and phrase boundaries require visual review. Not a clinical measure.'
    }


def save_results(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    if not rows:
        print('No recordings were successfully analyzed; no result files written.')
        return
    # Later recordings may add optional fields (for example, metronome-reference metrics).
    # Build a stable union so DictWriter can serialize heterogeneous result rows.
    fields=list(dict.fromkeys(key for row in rows for key in row.keys()))
    with (output/'rhythm_measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (output/'rhythm_measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # sqlite3.Connection's context manager commits/rolls back but does not
    # close the connection; close it explicitly to avoid ResourceWarnings.
    with closing(sqlite3.connect(output/'rhythm_history.sqlite')) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS analysis_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, analyzed_at_utc TEXT NOT NULL, analysis_version TEXT NOT NULL, filename TEXT NOT NULL, result_json TEXT NOT NULL)')
            now=datetime.now(timezone.utc).isoformat(timespec='seconds')
            db.executemany('INSERT INTO analysis_runs (analyzed_at_utc,analysis_version,filename,result_json) VALUES (?,?,?,?)',[(now,ANALYSIS_VERSION,r['filename'],json.dumps(r)) for r in rows])
    print(f'Results saved in {output.resolve()} (CSV, JSON, SQLite).')


def discover_files(source: Path, recursive: bool) -> list[Path]:
    if source.is_file(): return [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    it=source.rglob('*') if recursive else source.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def main() -> int:
    parser=argparse.ArgumentParser(description='Date-aware exploratory rhythm analysis for guitar audio/video.')
    parser.add_argument('input',type=Path,help='Media file or folder.')
    parser.add_argument('--recursive',action='store_true',help='Include nested folders.')
    parser.add_argument('--output',type=Path,default=Path('guitar_rhythm_results'))
    parser.add_argument('--min-duration',type=float,default=3.0,help='Minimum duration in seconds (default: 3).')
    parser.add_argument('--min-onsets',type=int,default=5,help='Onsets needed for a usable quality flag (default: 5).')
    parser.add_argument('--plots',action='store_true',help='Save waveform/onset diagnostic PNGs.')
    parser.add_argument('--mode',choices=['auto','self-paced','reference'],default='self-paced',help='Self-paced is the default; reference mode is retained for compatibility.')
    parser.add_argument('--reference-bpm',type=float,default=None,help='Known reference BPM.')
    args=parser.parse_args()
    if not args.input.exists():
        print(f'Input path does not exist: {args.input}',file=sys.stderr); return 2
    files=discover_files(args.input,args.recursive)
    if not files: print('No supported audio/video files found.'); return 1
    print(f'Found {len(files)} media file(s).')
    rows=[]; errors=[]
    plot_dir=args.output/'onset_plots' if args.plots else None
    for i,p in enumerate(files,1):
        print(f'[{i}/{len(files)}] {p}')
        try:
            r=analyze_recording(p,args.min_duration,args.min_onsets,plot_dir,args.mode,args.reference_bpm); rows.append(r)
            cv_text=f"{r['within_phrase_interval_cv_percent']:.2f}%" if r['within_phrase_interval_cv_percent'] is not None else 'n/a'
            print(f"  {r['detected_onsets']} onsets; {r['phrase_count']} phrases; within-phrase CV {cv_text}; quality {r['quality_flag']}")
        except Exception as e:
            errors.append({'file':str(p),'error':str(e),'exception_type':type(e).__name__,'traceback':traceback.format_exc()}); print(f'  SKIPPED: {e}',file=sys.stderr)
    save_results(rows,args.output)
    if errors:
        (args.output/'analysis_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        print(f'{len(errors)} file(s) skipped; see analysis_errors.json')
    else:
        stale_errors=args.output/'analysis_errors.json'
        if stale_errors.exists(): stale_errors.unlink()
    print('\nImportant: phrase boundaries and onset detections are heuristic. Inspect plots and compare similar playing styles; these measurements are not a validated clinical or motor assessment.')
    return 0 if rows else 1

if __name__=='__main__': raise SystemExit(main())
