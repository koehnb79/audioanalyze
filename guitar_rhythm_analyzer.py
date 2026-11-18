#!/usr/bin/env python3
"""Guitar Rhythm Analyzer 0.3.1: date-aware longitudinal rhythm analysis."""
from __future__ import annotations
import argparse, csv, json, math, re, shutil, sqlite3, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import librosa
import numpy as np

AUDIO_EXTENSIONS = {'.wav','.mp3','.flac','.m4a','.aac','.ogg','.aiff','.aif'}
VIDEO_EXTENSIONS = {'.mp4','.mov','.mkv','.avi','.webm','.mpeg','.mpg','.m4v'}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
ANALYSIS_VERSION = '0.4.0'


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
        if bpm and math.isfinite(bpm) and bpm>0:
            out.append({'start_seconds':round(start/sr,2),'end_seconds':round((start+win)/sr,2),'tempo_bpm':round(bpm,2),'beat_count':int(len(beats))})
    return out



def detect_rhythm_reference(y: np.ndarray, sr: int, guitar_onsets: np.ndarray) -> dict:
    """
    Conservative screening for a likely independent rhythmic layer.

    HPSS/percussive onset regularity is only indirect evidence: strummed guitar,
    palm muting, tapping, and other playing techniques can also appear percussive.
    Therefore this function never claims source identity and keeps ambiguous
    recordings eligible only as 'unknown' rather than asserting self-paced.
    """
    result = {
        'rhythm_reference_status': 'unknown',
        'rhythm_reference_type': 'unknown',
        'rhythm_reference_confidence': 'low',
        'percussive_onset_count': 0,
        'percussive_interval_cv_percent': None,
        'percussive_events_independent_of_guitar_percent': None,
        'percussive_activity_coverage_percent': None,
        'self_paced_eligible': None,
        'rhythm_reference_note': 'Insufficient evidence; manual review recommended.'
    }
    if len(y) < int(sr * 8):
        result['rhythm_reference_note'] = 'Recording too short for conservative reference screening.'
        return result
    try:
        harmonic, percussive = librosa.effects.hpss(y)
        hop = 256
        p_env = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=hop, aggregate=np.median)
        h_env = librosa.onset.onset_strength(y=harmonic, sr=sr, hop_length=hop, aggregate=np.median)
        p_frames = librosa.onset.onset_detect(onset_envelope=p_env, sr=sr, hop_length=hop,
                                               backtrack=False, units='frames', wait=2)
        h_frames = librosa.onset.onset_detect(onset_envelope=h_env, sr=sr, hop_length=hop,
                                               backtrack=False, units='frames', wait=2)
        p_times = np.asarray(librosa.frames_to_time(p_frames, sr=sr, hop_length=hop), dtype=float)
        h_times = np.asarray(librosa.frames_to_time(h_frames, sr=sr, hop_length=hop), dtype=float)
        if len(p_times):
            p_times = p_times[np.r_[True, np.diff(p_times) >= 0.08]]
        result['percussive_onset_count'] = int(len(p_times))
        if len(p_times) < 8:
            result['rhythm_reference_note'] = 'Too few distinct percussive transients for reliable screening.'
            return result
        intervals = np.diff(p_times)
        plausible = intervals[(intervals >= 0.25) & (intervals <= 1.5)]
        interval_cv = cv(plausible) if len(plausible) >= 6 else None
        result['percussive_interval_cv_percent'] = round(interval_cv, 2) if interval_cv is not None else None

        if len(h_times):
            distances = np.min(np.abs(p_times[:, None] - h_times[None, :]), axis=1)
            independent = distances > 0.075
        elif len(guitar_onsets):
            distances = np.min(np.abs(p_times[:, None] - guitar_onsets[None, :]), axis=1)
            independent = distances > 0.075
        else:
            independent = np.ones(len(p_times), dtype=bool)
        independent_pct = float(np.mean(independent) * 100)
        result['percussive_events_independent_of_guitar_percent'] = round(independent_pct, 2)
        duration = len(y) / sr
        coverage = float((p_times[-1] - p_times[0]) / duration * 100) if len(p_times) > 1 and duration else 0.0
        result['percussive_activity_coverage_percent'] = round(coverage, 2)

        # High confidence is deliberately difficult to reach. This is a review
        # flag, not proof that the percussion came from a separate backing track.
        if interval_cv is not None and interval_cv <= 10 and independent_pct >= 40 and coverage >= 55 and len(p_times) >= 12:
            result.update({
                'rhythm_reference_status': 'likely_external_rhythm',
                'rhythm_reference_type': 'regular_percussive_layer',
                'rhythm_reference_confidence': 'high',
                'self_paced_eligible': False,
                'rhythm_reference_note': 'Regular percussive events continue independently of many harmonic attacks; review audio before exclusion.'
            })
        elif interval_cv is not None and interval_cv <= 20 and independent_pct >= 25 and coverage >= 35:
            result.update({
                'rhythm_reference_status': 'possible_external_rhythm',
                'rhythm_reference_type': 'percussive_layer',
                'rhythm_reference_confidence': 'medium',
                'self_paced_eligible': None,
                'rhythm_reference_note': 'Some evidence of a regular independent percussive layer; manual review required.'
            })
        else:
            result.update({
                'rhythm_reference_status': 'no_strong_reference_evidence',
                'rhythm_reference_type': 'not_detected',
                'rhythm_reference_confidence': 'low',
                'self_paced_eligible': True,
                'rhythm_reference_note': 'No strong independent rhythmic-layer evidence detected; this does not prove the recording is self-paced.'
            })
    except Exception as exc:
        result['rhythm_reference_note'] = f'Reference screening failed: {exc}'
    return result


def plot_diagnostics(y: np.ndarray, sr: int, onsets: np.ndarray, phrases: list[np.ndarray], out: Path, stem: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out.mkdir(parents=True,exist_ok=True)
        times=np.arange(y.size)/sr
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,6),sharex=False,constrained_layout=True)
        step=max(1,int(sr/3000))
        ax1.plot(times[::step],y[::step],linewidth=.45)
        for t in onsets: ax1.axvline(t,alpha=.28,linewidth=.6)
        ax1.set(title='Waveform with detected onsets',xlabel='Time (s)',ylabel='Amplitude')
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


def pulse_analysis(y, sr, onsets, mode='self-paced', reference_bpm=None):
    """Estimate pulse regularity; do not fit an onset grid to the same performance."""
    hop=256
    env=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop,aggregate=np.median)
    bpm,beats=librosa.beat.beat_track(onset_envelope=env,sr=sr,hop_length=hop,units='time')
    bpm=float(np.asarray(bpm).reshape(-1)[0]) if np.asarray(bpm).size else None
    beats=np.asarray(beats,dtype=float)
    if reference_bpm is not None: bpm=float(reference_bpm); source='user_supplied_reference_bpm'
    else: source='self_paced_beat_tracker'
    return {'analysis_mode':mode,'pulse_source':source,'pulse_confidence':'tempo_estimate_only',
      'pulse_bpm':round(bpm,3) if bpm and math.isfinite(bpm) else None,
      'reference_candidate_bpm':None,'self_paced_candidate_bpm':round(bpm,3) if bpm and math.isfinite(bpm) else None,
      'pulse_times_seconds_json':json.dumps(beats.round(4).tolist()),
      'pulse_interval_cv_percent':cv(np.diff(beats)) if len(beats)>1 else None,
      'grid_subdivision':None,'onset_grid_median_abs_error_ms':None,
      'onset_grid_median_signed_error_ms':None,'onset_grid_p90_abs_error_ms':None,
      'onsets_within_50ms_percent':None,'grid_residuals_ms_json':'[]',
      'grid_metric_status':'disabled_in_0.3.1_no_independent_reference_phase'}


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
    recording_date_value,recording_date_source=recording_date(path)
    plot_path=plot_diagnostics(y,sr,onsets,phrases,plot_dir,path.stem) if plot_dir else None
    return {
      'filename':path.name,'path':str(path.resolve()),'media_type':'video' if path.suffix.lower() in VIDEO_EXTENSIONS else 'audio',
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
    fields=list(rows[0])
    with (output/'rhythm_measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (output/'rhythm_measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # Keep every recording in the master dataset, and emit transparent screening subsets.
    screening_fields = [k for k in fields if k in {
        'filename','recording_date','recording_year','duration_seconds',
        'rhythm_reference_status','rhythm_reference_type','rhythm_reference_confidence',
        'percussive_onset_count','percussive_interval_cv_percent',
        'percussive_events_independent_of_guitar_percent',
        'percussive_activity_coverage_percent','self_paced_eligible','rhythm_reference_note'
    }]
    screening = [{k:r.get(k) for k in screening_fields} for r in rows]
    with (output/'rhythm_reference_screening.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=screening_fields); w.writeheader(); w.writerows(screening)
    for name, predicate in [
        ('self_paced_candidates.csv', lambda r: r.get('self_paced_eligible') is True),
        ('external_reference_candidates.csv', lambda r: r.get('self_paced_eligible') is False),
        ('reference_review_required.csv', lambda r: r.get('self_paced_eligible') is None),
    ]:
        subset=[{k:r.get(k) for k in screening_fields} for r in rows if predicate(r)]
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=screening_fields); w.writeheader(); w.writerows(subset)
    # Keep every recording in the master dataset, and emit transparent screening subsets.
    screening_fields = [k for k in fields if k in {
        'filename','recording_date','recording_year','duration_seconds',
        'rhythm_reference_status','rhythm_reference_type','rhythm_reference_confidence',
        'percussive_onset_count','percussive_interval_cv_percent',
        'percussive_events_independent_of_guitar_percent',
        'percussive_activity_coverage_percent','self_paced_eligible','rhythm_reference_note'
    }]
    screening = [{k:r.get(k) for k in screening_fields} for r in rows]
    with (output/'rhythm_reference_screening.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=screening_fields); w.writeheader(); w.writerows(screening)
    for name, predicate in [
        ('self_paced_candidates.csv', lambda r: r.get('self_paced_eligible') is True),
        ('external_reference_candidates.csv', lambda r: r.get('self_paced_eligible') is False),
        ('reference_review_required.csv', lambda r: r.get('self_paced_eligible') is None),
    ]:
        subset=[{k:r.get(k) for k in screening_fields} for r in rows if predicate(r)]
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=screening_fields); w.writeheader(); w.writerows(subset)
    with sqlite3.connect(output/'rhythm_history.sqlite') as db:
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
            errors.append({'file':str(p),'error':str(e)}); print(f'  SKIPPED: {e}',file=sys.stderr)
    save_results(rows,args.output)
    if errors:
        (args.output/'analysis_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        print(f'{len(errors)} file(s) skipped; see analysis_errors.json')
    print('\nImportant: phrase boundaries and onset detections are heuristic. Inspect plots and compare similar playing styles; these measurements are not a validated clinical or motor assessment.')
    return 0 if rows else 1

if __name__=='__main__': raise SystemExit(main())
