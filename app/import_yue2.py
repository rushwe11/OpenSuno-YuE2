#!/usr/bin/env python3
"""Import songs from an existing YuE2 Atelier project into OpenSuno's library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def job_id(source: Path) -> str:
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime((source / 'audio.flac').stat().st_mtime))
    return stamp + '-' + source_key(source)[:8]


def request_settings(source: Path, result: dict, external_root: Path) -> dict:
    request = read_json(source / 'request.json', {}) or {}
    config = read_json(source / 'config.json', {}) or {}
    generation = config.get('generation') if isinstance(config.get('generation'), dict) else {}
    abc = generation.get('abc') if isinstance(generation.get('abc'), dict) else {
        'temperature': .7, 'top_p': .9, 'top_k': 30, 'repetition_penalty': 1.005,
        'penalty_window': 100, 'min_tokens': 32, 'max_tokens': 4096,
    }
    semantic = generation.get('semantic') if isinstance(generation.get('semantic'), dict) else {
        'temperature': 1., 'top_p': .95, 'top_k': 100, 'repetition_penalty': 1.2,
        'penalty_window': 50, 'min_tokens': 200,
        'max_tokens': max(200, int(round(float(result.get('audio_seconds', 0)) * 25))),
    }
    studio = read_json(source / 'studio.json', {}) or {}
    title = studio.get('title') or str(request.get('id') or source.name).replace('-', ' ').replace('_', ' ').title()
    cot = request.get('cot') if request.get('cot') in {'full', 'melody', 'off'} else 'off'
    cfg = request.get('cfg_scale')
    if cfg is None:
        cfg = 1.01 if cot == 'off' else 1.0
    return {
        'kind': 'generate', 'model': 'mps-pytorch', 'title': title[:100],
        'style': str(request.get('style') or request.get('tags') or ''),
        'lyrics': str(request.get('lyrics') or ''), 'instrumental': False,
        'voice': 'any', 'cot': cot, 'abc': '', 'seed': int(request.get('seed') or 0),
        'random_seed': False, 'candidates': 1, 'cfg_scale': float(cfg),
        'steps': int(generation.get('ode_steps') or studio.get('steps') or 2),
        'abc_sampling': abc, 'semantic_sampling': semantic,
        'workspace_id': 'default', 'lora': '', 'lora_strength': 1.0,
        'sound_lora': '', 'sound_lora_strength': 1.0,
        'origin': {'kind': 'atelier-import', 'source': str(source.resolve()),
                   'project': str(external_root.resolve())},
    }


def write_wav(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix('.tmp')
    with sf.SoundFile(source, 'r') as reader:
        with sf.SoundFile(temporary, 'w', samplerate=reader.samplerate, channels=reader.channels,
                          format='WAV', subtype='PCM_16') as writer:
            for block in reader.blocks(blocksize=65536, dtype='float32', always_2d=True):
                writer.write(block)
    temporary.replace(destination)


def copy_optional(source: Path, destination: Path, name: str) -> None:
    path = source / name
    if path.is_file():
        shutil.copy2(path, destination / name)


def import_song(source: Path, library: Path, external_root: Path) -> tuple[str, bool]:
    result = read_json(source / 'result.json')
    if not isinstance(result, dict) or result.get('status') != 'complete' or not (source / 'audio.flac').is_file():
        raise ValueError('Not a completed YuE2 song: ' + str(source))
    identifier = job_id(source)
    destination = library / identifier
    imported = read_json(destination / 'import-source.json', {}) if destination.exists() else {}
    if imported and imported.get('source_key') == source_key(source):
        return identifier, False
    if destination.exists():
        raise FileExistsError('Library destination already exists: ' + str(destination))
    destination.mkdir(parents=True)
    try:
        settings = request_settings(source, result, external_root)
        write_wav(source / 'audio.flac', destination / 'audio.wav')
        score = (source / 'score.abc').read_text(encoding='utf-8') if (source / 'score.abc').is_file() else ''
        if score:
            (destination / 'score.abc').write_text(score, encoding='utf-8')
        semantic = np.load(source / 'semantic.npy', allow_pickle=False)
        if semantic.ndim != 1 or semantic.dtype.kind not in 'iu':
            raise ValueError('Invalid semantic.npy in ' + str(source))
        tokens = [int(value) for value in semantic.tolist()]
        atomic_json(destination / 'tokens.json', tokens)
        prefix_path = source / 'prefix.npy'
        if prefix_path.is_file():
            prefix = np.load(prefix_path, allow_pickle=False)
            if prefix.ndim != 1 or prefix.dtype.kind not in 'iu':
                raise ValueError('Invalid prefix.npy in ' + str(source))
            atomic_json(destination / 'render-prefix.json', [int(value) for value in prefix.tolist()])
        atomic_json(destination / 'settings.json', settings)
        atomic_json(destination / 'request.json', settings)
        seconds = float(result.get('audio_seconds') or len(tokens) / 25)
        elapsed = float((result.get('timing') or {}).get('e2e_seconds') or 0)
        candidate = {'index': 1, 'prefix': '', 'seed': settings['seed'], 'model': 'mps-pytorch',
                     'title': settings['title'], 'seconds': seconds, 'elapsed': elapsed,
                     'abc': score, 'tokens': len(tokens)}
        open_result = {'model': 'mps-pytorch', 'seconds': seconds, 'elapsed': elapsed,
                       'seed': settings['seed'], 'abc': score, 'imported': True,
                       'candidates': [candidate]}
        atomic_json(destination / 'result.json', open_result)
        atomic_json(destination / 'state.json', {'status': 'complete', 'updated': time.time()})
        atomic_json(destination / 'import-source.json', {
            'source': str(source.resolve()), 'source_key': source_key(source),
            'result_identity': result.get('identity'), 'weights': result.get('weights'),
        })
        for name in ('latent.npy', 'quality.json', 'studio.json'):
            copy_optional(source, destination, name)
        return identifier, True
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def discover(external_root: Path):
    runs = external_root / 'runs'
    if not runs.is_dir():
        return []
    return sorted({path.parent for path in runs.rglob('result.json')
                   if '.studio-jobs' not in path.parts and (path.parent / 'audio.flac').is_file()})


def import_all(external_root: Path, library: Path):
    library.mkdir(parents=True, exist_ok=True)
    report = {'imported': [], 'existing': [], 'skipped': []}
    for source in discover(external_root):
        try:
            identifier, created = import_song(source, library, external_root)
            report['imported' if created else 'existing'].append({'id': identifier, 'source': str(source)})
        except (ValueError, OSError) as error:
            report['skipped'].append({'source': str(source), 'error': str(error)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing YuE2 project root')
    parser.add_argument('--library', type=Path, required=True, help='OpenSuno library folder')
    args = parser.parse_args()
    print(json.dumps(import_all(args.source.resolve(), args.library.resolve()), indent=2))


if __name__ == '__main__':
    main()
