#!/usr/bin/env python3
# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""The Fax Test: can a language model write a whole TIFF decoder, CCITT fax codes included, in one answer?

One request (prompt.md, no tools) goes to any OpenAI-compatible chat API. The Python module in the answer is run on
40 fixed TIFF files (corpus/), each in its own process inside a sandbox, and a file counts only if every pixel is
right. With --turns N the model then sees which files failed and why, and writes the module again, up to N times.

  python3 faxtest.py run --model MODEL [--base-url URL]   # the key, if any, from OPENAI_API_KEY
  python3 faxtest.py grade decoder.py                      # grade a module written anywhere
  python3 faxtest.py selftest                              # the grader on decoders with known scores
  python3 faxtest.py table                                 # a Markdown table of your saved runs

Standard library only, Python 3.10 or newer. The sandbox is sandbox-exec on macOS and a throwaway Docker container
elsewhere (--sandbox); see README.md.
"""
import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sysconfig
import textwrap
import threading
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, 'corpus')
PROMPT = os.path.join(HERE, 'prompt.md')
CHECK = os.path.join(HERE, 'check')
RESULTS = os.path.join(HERE, 'results')
WORK = os.path.join(HERE, '.work')          # Docker's work folders: inside the clone, which Docker can mount
DOCKER_IMAGE = 'python:3.12-alpine'
CATEGORIES = ('uncompressed', 'packbits', 'lzw', 'deflate', 'ccitt_rle', 'g3_1d', 'g3_2d', 'g4')
CCITT = ('ccitt_rle', 'g3_1d', 'g3_2d', 'g4')
# what the prompt forbids, looked for in the code (the sandbox also hides third-party packages)
FORBIDDEN = re.compile(r'\b(subprocess|ctypes|cffi|os\.system|os\.popen|os\.exec\w*|os\.spawn\w*|PIL|numpy|'
                       r'tifffile|imageio|cv2|libtiff)\b')

OUT_LIMIT = 64 << 20          # bytes a module may write to any file, its output included
MEM_LIMIT = 1536 << 20        # resident memory before the module is stopped
SAFE_ENV = {'PATH': '/usr/bin:/bin'}   # all a sandboxed module gets of your environment: no keys, no home

RUNNER = r'''
import json, os, resource, sys, threading, time
FSIZE, MEM, CPU = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE, FSIZE))
resource.setrlimit(resource.RLIMIT_CPU, (CPU, CPU))
SCALE = 1 if sys.platform == 'darwin' else 1024          # ru_maxrss: bytes on macOS, kilobytes on Linux


def watch():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * SCALE > MEM:
            os.write(2, b'MemoryError: the module used more than the memory limit\n')
            os._exit(137)
        time.sleep(0.05)


threading.Thread(target=watch, daemon=True).start()
sys.path.insert(0, '.')
t0 = time.perf_counter()
import submission
t1 = time.perf_counter()
data = open(sys.argv[1], 'rb').read()
w, h, c, px = submission.decode(data)
t2 = time.perf_counter()
if not all(type(v) is int for v in (w, h, c)):
    raise TypeError('decode() must return int width, height and channels')
px = bytes(px)
with open(sys.argv[2], 'wb') as f:
    f.write(json.dumps({'width': w, 'height': h, 'channels': c, 'import_s': t1 - t0, 'decode_s': t2 - t1}).encode()
            + b'\n' + px)
'''


# ---- the sandbox ----

RUN_ID = uuid.uuid4().hex[:12]   # labels this run's containers, so an interrupted run can stop them


def pick_sandbox(choice='auto'):
    """'sandbox-exec' (macOS), 'docker' or 'none'. auto: sandbox-exec on macOS, else Docker if it answers."""
    if choice in ('sandbox-exec', 'docker', 'none'):
        return choice
    if sys.platform == 'darwin' and shutil.which('sandbox-exec'):
        return 'sandbox-exec'
    if shutil.which('docker'):
        try:
            if subprocess.run(['docker', 'info'], capture_output=True, timeout=20).returncode == 0:
                return 'docker'
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit('No sandbox found: on macOS sandbox-exec is built in; elsewhere install and start Docker. '
                     'To run the model\'s code unconfined on your machine, pass --sandbox none (not recommended).')


def sb_path(p):
    """A path as a quoted string in a sandbox profile."""
    return '"' + os.path.realpath(p).replace('\\', '\\\\').replace('"', '\\"') + '"'


def macos_profile(tmp, allow_site=False):
    """sandbox-exec rules, everything denied unless allowed: start this Python, read the system's libraries, Python's
    standard library and its own folder, write only there. No network, no other program, no child process, no
    system services, nothing of your files; third-party packages hidden unless allow_site (the selftest's oracle)."""
    exe = sys.executable
    py = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, os.path.dirname(os.path.realpath(exe)))})
    site = sorted({os.path.realpath(sysconfig.get_paths()[k]) for k in ('purelib', 'platlib')})
    hide = '' if allow_site else '(deny file-read* ' + ' '.join(f'(subpath {sb_path(p)})' for p in site) + ')'
    return f"""(version 1)
(deny default)
(allow process-exec (literal {sb_path(exe)}) (literal {sb_path(os.path.realpath(exe))}))
(allow file-read-metadata)
(allow file-read* (literal "/") (subpath "/System") (subpath "/usr/lib") (subpath "/private/var/db/dyld")
  (literal "/dev/urandom") (literal "/dev/null") (subpath {sb_path(tmp)}) {' '.join(f'(subpath {sb_path(p)})' for p in py)})
{hide}
(allow file-write* (subpath {sb_path(tmp)}) (literal "/dev/null"))
(allow sysctl-read)
"""


def ensure_image():
    if subprocess.run(['docker', 'image', 'inspect', DOCKER_IMAGE], capture_output=True).returncode != 0:
        print(f'pulling {DOCKER_IMAGE} (once) ...', file=sys.stderr)
        subprocess.run(['docker', 'pull', '-q', DOCKER_IMAGE], check=True)


def stop_containers():
    ids = subprocess.run(['docker', 'ps', '-q', '--filter', f'label=faxtest-run={RUN_ID}'],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(['docker', 'kill', *ids], capture_output=True)


def read_output(path):
    """out.bin as written by the runner: a regular file, not a link, at most OUT_LIMIT bytes."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError:
        raise RuntimeError('no output: the module ended before decode() returned')
    except OSError as e:
        raise RuntimeError(f'output not readable: {e.strerror}')
    with os.fdopen(fd, 'rb') as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise RuntimeError('output is not a regular file')
        data = f.read(OUT_LIMIT + 1)
    if len(data) > OUT_LIMIT:
        raise RuntimeError('output larger than the limit')
    head, _, px = data.partition(b'\n')
    try:
        head = json.loads(head)
    except ValueError:
        raise RuntimeError('output header unreadable')
    if not isinstance(head, dict) or not all(type(head.get(k)) is int for k in ('width', 'height', 'channels')):
        raise RuntimeError('decode() must return int width, height and channels')
    return head, px


def last_error(path):
    """The last line the module wrote to stderr, at most 300 characters, read from the end of the file."""
    try:
        with open(path, 'rb') as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - 4096))
            tail = f.read().decode('utf-8', 'replace')
    except OSError:
        return ''
    lines = [ln for ln in tail.strip().splitlines() if ln.strip() and 'MallocStackLogging' not in ln]
    return lines[-1][:300] if lines else ''


def run_one(code_path, tif_path, timeout=60, sandbox='sandbox-exec', site=False, env=None):
    """The module on one file, in a fresh process: (header dict, pixel bytes), or RuntimeError(reason)."""
    if sandbox == 'docker':
        os.makedirs(WORK, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='faxtest-', dir=WORK if sandbox == 'docker' else None)
    try:
        shutil.copy(code_path, os.path.join(tmp, 'submission.py'))
        shutil.copy(tif_path, os.path.join(tmp, 'in.tif'))
        with open(os.path.join(tmp, 'runner.py'), 'w', encoding='utf-8') as f:
            f.write(RUNNER)
        limits = [str(OUT_LIMIT), str(MEM_LIMIT), str(timeout)]
        py_args = ['-I', '-B', '-X', 'utf8'] + ([] if site else ['-S']) + ['runner.py', 'in.tif', 'out.bin'] + limits
        if sandbox == 'docker':
            user = ['--user', f'{os.getuid()}:{os.getgid()}'] if hasattr(os, 'getuid') else []
            envs = [a for k, v in (env or {}).items() for a in ('-e', f'{k}={v}')]
            cmd = ['docker', 'run', '--rm', '--label', f'faxtest-run={RUN_ID}', '--network', 'none',
                   '--memory', '1g', '--memory-swap', '1g', '--cpus', '1', '--pids-limit', '64', '--read-only',
                   '--tmpfs', '/tmp:size=64m', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', *user,
                   *envs, '-v', f'{tmp}:/w', '-w', '/w', DOCKER_IMAGE,
                   'timeout', '-s', 'KILL', str(timeout), 'python', *py_args]
            child_env, wait = os.environ, timeout + 30           # the Docker client's own settings; the start
        else:
            cmd = [sys.executable, *py_args]
            if sandbox == 'sandbox-exec':
                cmd = ['sandbox-exec', '-p', macos_profile(tmp, allow_site=site)] + cmd
            child_env, wait = {**SAFE_ENV, **(env or {})}, timeout + 5
        with open(os.path.join(tmp, 'stderr.txt'), 'wb') as err:
            try:
                p = subprocess.run(cmd, cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                   timeout=wait, env=child_env)
            except subprocess.TimeoutExpired:
                if sandbox == 'docker':
                    stop_containers()
                raise RuntimeError(f'timeout after {timeout} s')
        if p.returncode != 0:
            if p.returncode in (-9, -24, 137, 152) and not last_error(os.path.join(tmp, 'stderr.txt')):
                raise RuntimeError(f'timeout after {timeout} s')   # killed, or out of CPU time
            raise RuntimeError(last_error(os.path.join(tmp, 'stderr.txt')) or f'exit {p.returncode}')
        return read_output(os.path.join(tmp, 'out.bin'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- grading ----

def check_one(code_path, name, meta, timeout, sandbox, site):
    """One file's result: status pass | wrong | shape | error | timeout, with how close it came."""
    exp = zlib.decompress(open(os.path.join(CORPUS, name + '.expected.z'), 'rb').read())
    t0 = time.time()
    try:
        head, px = run_one(code_path, os.path.join(CORPUS, name + '.tif'), timeout, sandbox, site)
    except Exception as e:   # anything one module does stays with its file
        msg = str(e) if isinstance(e, RuntimeError) else f'{type(e).__name__}: {e}'
        return {'status': 'timeout' if msg.startswith('timeout') else 'error', 'error': msg[:300],
                'seconds': round(time.time() - t0, 2)}
    r = {'seconds': round(time.time() - t0, 2), 'decode_s': round(float(head.get('decode_s') or 0), 3),
         'got': [head['width'], head['height'], head['channels'], len(px)]}
    want = (meta['width'], meta['height'], meta['channels'])
    if tuple(r['got'][:3]) != want or len(px) != len(exp):
        r.update(status='shape', want=list(want) + [len(exp)])
        return r
    c = meta['channels']
    good = sum(px[i:i + c] == exp[i:i + c] for i in range(0, len(exp), c)) if px != exp else len(exp) // c
    r.update(status='pass' if px == exp else 'wrong', pixels_right=round(good / (len(exp) // c), 6))
    return r


def summarize(files, manifest):
    by = {}
    for name, r in files.items():
        b = by.setdefault(manifest[name]['category'], [0, 0])
        b[0] += r['status'] == 'pass'
        b[1] += 1
    return {'passed': sum(v[0] for v in by.values()), 'of': len(files),
            'ccitt_passed': sum(by.get(c, [0, 0])[0] for c in CCITT),
            'ccitt_of': sum(by.get(c, [0, 0])[1] for c in CCITT),
            'by_category': {c: by[c] for c in CATEGORIES if c in by},
            'statuses': {s: sum(r['status'] == s for r in files.values())
                         for s in ('pass', 'wrong', 'shape', 'error', 'timeout')}}


def grade(code_path, jobs=4, timeout=60, sandbox='sandbox-exec', site=False, echo=None):
    """Runs the module at code_path on every corpus file: {'files': {name: result}, 'summary': {...}}."""
    manifest = json.load(open(os.path.join(CORPUS, 'manifest.json'), encoding='utf-8'))['files']
    if sandbox == 'docker':
        ensure_image()
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
            futs = {name: ex.submit(check_one, code_path, name, meta, timeout, sandbox, site)
                    for name, meta in manifest.items()}
            files = {name: f.result() for name, f in futs.items()}
    except KeyboardInterrupt:
        if sandbox == 'docker':
            stop_containers()
        raise
    if echo:
        for name, r in files.items():
            extra = r.get('error') or (f"{r['pixels_right']:.2%} of pixels right" if r['status'] == 'wrong' else '') \
                or (f"got {r['got']}, want {r['want']}" if r['status'] == 'shape' else '')
            echo(f"  {name:34s} {r['status']:7s} {r['seconds']:6.2f} s  {extra}")
    s = summarize(files, manifest)
    s['grading_s'] = round(time.time() - t0, 1)
    s['sandbox'] = sandbox
    s['forbidden_words'] = sorted(set(FORBIDDEN.findall(open(code_path, encoding='utf-8', errors='replace').read())))
    return {'files': files, 'summary': s}


def summary_line(s):
    cats = '  '.join(f'{c} {p}/{n}' for c, (p, n) in s['by_category'].items())
    return f"{s['passed']}/{s['of']} files (CCITT {s['ccitt_passed']}/{s['ccitt_of']})   {cats}"


# ---- the request ----

def chat(base, key, body, timeout=900, progress=print):
    """One streamed chat request: the message (content and any reasoning), the finish reason ('cut-off' when the
    stream ended without one), and its timing. timeout: seconds without a byte from the server."""
    headers = {'Content-Type': 'application/json', 'User-Agent': 'fax-test/1'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions', json.dumps({
        **body, 'stream': True, 'stream_options': {'include_usage': True}}).encode(), headers)
    t0, first, usage, finish, done, shown = time.time(), None, {}, None, False, time.time()
    msg = {'content': '', 'reasoning': ''}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                continue
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get('error'):
                raise RuntimeError(f"server error: {str(ev['error'])[:300]}")
            usage = ev.get('usage') or usage
            for ch in ev.get('choices') or []:
                finish = ch.get('finish_reason') or finish
                d = ch.get('delta') or {}
                text, thought = d.get('content') or '', d.get('reasoning') or d.get('reasoning_content') or ''
                if (text or thought) and first is None:
                    first = time.time()
                msg['content'] += text
                msg['reasoning'] += thought
            if time.time() - shown > 30:
                shown = time.time()
                progress(f"    ... {time.time() - t0:.0f} s, {len(msg['reasoning']):,} characters of reasoning, "
                         f"{len(msg['content']):,} of answer")
    t1 = time.time()
    n_out = usage.get('completion_tokens') or 0
    gen = t1 - (first or t1)
    return msg, finish or ('stop' if done else 'cut-off'), {
        'seconds': round(t1 - t0, 1), 'first_token_s': round((first or t1) - t0, 2),
        'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': n_out,
        'tokens_per_s': round(n_out / gen, 1) if gen > 0.05 and n_out > 1 else None}


def extract_code(text):
    """The module from an answer: the longest fenced code block that defines decode(), Python-tagged or untagged
    ones first; fences may be indented, tagged in any case, ``` or ~~~; an unclosed last block (an answer cut off by
    the token limit) counts up to the end. An answer that is only code counts whole. None when there is none."""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    blocks, cur, lang, fence = [], None, '', ''
    for ln in lines:
        st = ln.strip()
        if cur is None:
            m = re.match(r'(`{3,}|~{3,})\s*([\w+.-]*)', st)
            if m:
                fence, lang, cur = m.group(1), m.group(2).lower(), []
        elif re.fullmatch(re.escape(fence[0]) + '{%d,}' % len(fence), st):
            blocks.append((lang, '\n'.join(cur)))
            cur = None
        else:
            cur.append(ln)
    if cur is not None:
        blocks.append((lang, '\n'.join(cur)))
    defines = lambda b: re.search(r'^\s*def decode\s*\(', b, re.M)  # noqa: E731
    good = [b for t, b in blocks if t in ('', 'python', 'py', 'python3') and defines(b)] or \
        [b for t, b in blocks if defines(b)]
    if not good and not blocks and re.search(r'^def decode\s*\(', text, re.M):
        good = [text]
    return textwrap.dedent(max(good, key=len)) if good else None


def feedback_text(g, manifest):
    """What the model is told after a turn, as a test runner prints it: the files that passed, and for each failed
    file its TIFF tags and what went wrong. Never the files or their pixels."""
    if g is None:
        return ('Your answer contained no Python module defining decode(), so no file could be tested. Reply with the '
                'complete module in a single ```python code block.')
    s, files = g['summary'], g['files']
    lines = [f"Test results: {s['passed']} of {s['of']} files decoded exactly."]
    passed = [n for n, r in files.items() if r['status'] == 'pass']
    if passed:
        lines.append('Passed: ' + ', '.join(passed) + '.')
    lines.append('Failed:')
    for n, r in files.items():
        if r['status'] == 'pass':
            continue
        if r['status'] in ('error', 'timeout'):
            what = r.get('error') if r['status'] == 'timeout' else f"raised {r.get('error')}"
        elif r['status'] == 'shape':
            what = f"returned (width, height, channels, len(pixels)) = {tuple(r['got'])}, expected {tuple(r['want'])}"
        else:
            what = f"right size, {r['pixels_right']:.2%} of pixels right"
        lines.append(f"- {n} ({manifest[n]['tags']}): {what}")
    lines.append('Fix the decoder. Reply with the complete corrected module in a single ```python code block.')
    return '\n'.join(lines)


def ask_and_grade(base, key, body, *, timeout, jobs, sandbox, echo):
    try:
        msg, finish, m = chat(base, key, body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {'error': f'HTTP {e.code} {e.read().decode(errors="replace")[:300]}'}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as e:
        return {'error': f'{type(e).__name__}: {e}'}
    turn = {'max_tokens': body.get('max_tokens'), 'finish': finish, 'timing': m, 'answer': msg['content'],
            'reasoning': msg['reasoning'], 'code': extract_code(msg['content']), 'grade': None}
    echo(f"  {m['seconds']:.0f} s, {m['completion_tokens']} tokens out, finish {finish}; "
         f"answer {len(msg['content'])} characters, reasoning {len(msg['reasoning'])}")
    if finish == 'cut-off':
        echo('  the stream ended without a finish reason: the answer may be incomplete')
    if finish == 'length' and turn['code'] is None:
        echo('  the token limit ended the answer before any code (raise --max-tokens for a reasoning model)')
    if turn['code'] is None:
        echo('  no module with decode() in the answer: 0 files')
        return turn
    turn['code_lines'] = turn['code'].count('\n') + 1
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(turn['code'])
    try:
        turn['grade'] = grade(f.name, jobs=jobs, sandbox=sandbox, echo=echo)
    finally:
        os.unlink(f.name)
    echo(f"  {turn['code_lines']} lines: {summary_line(turn['grade']['summary'])}")
    return turn


def run_test(base, key, model, *, max_tokens=32768, extra=None, turns=0, context=None, timeout=900, jobs=4,
             sandbox='sandbox-exec', echo=print, save=None):
    """The first request and its grade (the one-shot score), then up to `turns` more, each told its results, until
    every file passes. context: the model's tokens per request, so a later turn's max_tokens fits the conversation.
    max_tokens 0 sends no limit (for APIs that want it under another name in extra). save(record) runs after each
    turn. Returns the record: the first turn at the top level, later turns in record['feedback']."""
    prompt = open(PROMPT, encoding='utf-8').read()
    manifest = json.load(open(os.path.join(CORPUS, 'manifest.json'), encoding='utf-8'))['files']
    messages = [{'role': 'user', 'content': prompt}]
    rec = {'bench': 'fax-test', 'version': 1, 'model': model, 'api_host': urllib.parse.urlparse(base).hostname,
           'started': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'settings': {'max_tokens': max_tokens, 'extra': extra, 'turns': turns, 'context': context,
                        'sandbox': sandbox}, 'prompt': prompt, 'feedback': []}
    echo(f'{model}: max_tokens {max_tokens}, extra fields {extra or "none"}, {turns} feedback turns')
    prompt_tokens = None
    for n in range(turns + 1):
        budget = max_tokens
        if context and max_tokens > 0:   # the prompt so far (measured, or estimated at 2.5 characters a token)
            if prompt_tokens:
                used = prompt_tokens + sum(len(x['content']) for x in messages[-2:]) * 2 // 5
            else:
                used = sum(len(x['content']) for x in messages) * 2 // 5
            budget = min(max_tokens, context - used - 1024)
            if budget < 4000:
                echo(f'  turn {n + 1}: the conversation leaves {budget} tokens of {context}: stopped')
                rec['stopped'] = 'context full'
                break
        body = {'model': model, 'messages': messages, **({'max_tokens': budget} if budget > 0 else {}), **(extra or {})}
        echo(f'  turn {n + 1} of {turns + 1}:')
        turn = ask_and_grade(base, key, body, timeout=timeout, jobs=jobs, sandbox=sandbox, echo=echo)
        (rec.update(turn) if n == 0 else rec['feedback'].append(turn))
        if save:
            save(rec)
        if 'error' in turn:
            echo('  ' + turn['error'])
            break
        prompt_tokens = turn['timing']['prompt_tokens']
        g = turn['grade']
        if g and g['summary']['passed'] == g['summary']['of']:
            break
        messages = messages + [{'role': 'assistant', 'content': turn['answer']},
                               {'role': 'user', 'content': feedback_text(g, manifest)}]
    done = [t for t in [rec] + rec['feedback'] if 'error' not in t]
    rec['final'] = {'passed_by_turn': [(t.get('grade') or {}).get('summary', {}).get('passed', 0) for t in done],
                    'ccitt_by_turn': [(t.get('grade') or {}).get('summary', {}).get('ccitt_passed', 0) for t in done],
                    'tokens_out': sum(t['timing']['completion_tokens'] for t in done)}
    if save:
        save(rec)
    return rec


# ---- the grader's own check ----

OFF_BY_ONE = '''

_right = decode


def decode(data):
    w, h, c, px = _right(data)
    px = bytearray(px)
    px[len(px) // 2] ^= 1
    return w, h, c, bytes(px)
'''
STRIPS_ONLY = {'raw_gray8', 'raw_rgb8', 'raw_rgb8_strips_be', 'deflate_gray8', 'deflate_rgb8_predictor',
               'deflate_old_code_rgb8'}   # what check/strips_only.py handles: 8-bit gray/RGB strips, none/Deflate
PROBE_BITS = ('read this folder', 'connect', 'child process', 'write outside its folder', 'start a program',
              'read /etc/hosts', 'import an installed package')
# what a module may do inside each sandbox: inside a throwaway container, processes and the container's own /etc
# reach nothing of yours (bits 4, 16, 32); under sandbox-exec, nothing
PROBE_ALLOWED = {'sandbox-exec': 0, 'docker': 4 | 16 | 32}


def site_package():
    """(site-packages folder, a package installed there) for the probe, or ('', '')."""
    for d in sysconfig.get_paths()['purelib'], sysconfig.get_paths()['platlib']:
        for name in ('pip', 'setuptools', 'PIL', 'numpy'):
            if os.path.isdir(os.path.join(d, name)):
                return d, name
    return '', ''


def selftest(sandbox, jobs=4):
    """Decoders whose scores are known before they run (check/): the grader must give exactly those; then what the
    sandbox lets a module do."""
    manifest = json.load(open(os.path.join(CORPUS, 'manifest.json'), encoding='utf-8'))['files']
    n, ok = len(manifest), True
    have_pil = False
    if sandbox != 'docker':   # Pillow as the sandboxed module would see it (site packages allowed for the oracle)
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('from PIL import Image\n\n\ndef decode(data):\n    return 1, 1, 1, b"\\x00"\n')
        try:
            run_one(f.name, os.path.join(CORPUS, 'raw_gray8.tif'), sandbox=sandbox, site=True)
            have_pil = True
        except RuntimeError:
            pass
        finally:
            os.unlink(f.name)
    oracle = os.path.join(CHECK, 'oracle_pillow.py')
    off = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8')
    off.write(open(oracle, encoding='utf-8').read() + OFF_BY_ONE)
    off.close()
    cases = [  # (file, site packages, expected passes, every other file's status, what it shows)
        (os.path.join(CHECK, 'blank.py'), False, set(), 'shape', 'a 1x1 white image: every file the wrong size'),
        (os.path.join(CHECK, 'strips_only.py'), False, STRIPS_ONLY, None,
         'a 40-line decoder for 8-bit strips, none or Deflate: exactly those 6 files'),
    ]
    if have_pil:
        cases = [(oracle, True, set(manifest), None, 'libtiff through Pillow: every file'),
                 (off.name, True, set(), 'wrong', 'the same with one sample changed per image: every file wrong')] + cases
    else:
        print('skipped the Pillow oracle (Pillow is not installed for this Python, or the sandbox is Docker)')
    try:
        for path, site, want, others, what in cases:
            g = grade(path, jobs=jobs, sandbox=sandbox, site=site)
            got = {k for k, r in g['files'].items() if r['status'] == 'pass'}
            bad = {k for k, r in g['files'].items() if others and k not in want and r['status'] != others}
            good = got == want and not bad
            ok &= good
            label = 'oracle_one_pixel_off' if path == off.name else os.path.basename(path)
            print(f"{'ok ' if good else 'BAD'} {label:24s} {len(got):2d}/{n} pass, want {len(want)}: {what}")
            if not good:
                print(f'    extra: {sorted(got - want)}\n    missing: {sorted(want - got)}\n    not {others}: {sorted(bad)[:5]}')
    finally:
        os.unlink(off.name)
    if sandbox == 'none':
        print('no sandbox: its check is skipped')
    else:
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(4)
        site_dir, package = site_package()
        try:
            _, px = run_one(os.path.join(CHECK, 'escape_probe.py'), os.path.join(CORPUS, 'raw_gray8.tif'),
                            sandbox=sandbox, env={'FAXTEST_PROBE': os.path.join(HERE, 'README.md'),
                                                  'FAXTEST_PORT': str(listener.getsockname()[1]),
                                                  'FAXTEST_SITE': site_dir, 'FAXTEST_PACKAGE': package})
        finally:
            listener.close()
        flags, allowed = px[0], PROBE_ALLOWED[sandbox]
        good = flags & ~allowed == 0
        ok &= good
        done = ', '.join(f"{w} {'yes' if flags & (1 << i) else 'no'}" for i, w in enumerate(PROBE_BITS))
        print(f"{'ok ' if good else 'BAD'} escape_probe.py          {done}"
              + (' (yes is allowed inside the container for child process, start a program, read /etc/hosts)'
                 if sandbox == 'docker' else ''))
    print(f"selftest {'passed' if ok else 'FAILED'} (sandbox: {sandbox})")
    return ok


# ---- saved runs ----

def table(paths):
    head = ['model', 'started (UTC)', 'turns', 'tokens out', 'files by turn', 'best', 'CCITT best', *CATEGORIES]
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head))
    for p in paths:
        r = json.load(open(p, encoding='utf-8'))
        if r.get('bench') != 'fax-test':
            continue
        turns = [t for t in [r] + r.get('feedback', []) if 'error' not in t]
        grades = [t['grade'] for t in turns if t.get('grade')]
        best = max(grades, key=lambda g: g['summary']['passed'], default=None)
        cats = best['summary']['by_category'] if best else {}
        print('| ' + ' | '.join([r['model'], r['started'][:16].replace('T', ' '), str(len(turns)),
                                 str(r.get('final', {}).get('tokens_out', '')),
                                 ' → '.join(map(str, r.get('final', {}).get('passed_by_turn', []))) or r.get('error', '')[:40],
                                 f"{best['summary']['passed'] if best else 0}/40",
                                 f"{max((g['summary']['ccitt_passed'] for g in grades), default=0)}/19"] +
                                [f'{cats[c][0]}/{cats[c][1]}' if c in cats else '-' for c in CATEGORIES]) + ' |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='ask a model and grade its answer')
    r.add_argument('--model', required=True, help='the model name the API expects')
    r.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'),
                   help='the API base ending in /v1 (default: $OPENAI_BASE_URL), e.g. http://localhost:11434/v1')
    r.add_argument('--api-key-env', default='OPENAI_API_KEY', help='the variable holding the key (local servers need none)')
    r.add_argument('--max-tokens', type=int, default=32768,
                   help='output tokens per request, reasoning included; 0 sends no limit (then use --extra)')
    r.add_argument('--turns', type=int, default=0, help='feedback turns after the first answer (default 0: one-shot)')
    r.add_argument('--context', type=int, help="the model's tokens per request, so later turns fit")
    r.add_argument('--temperature', type=float)
    r.add_argument('--extra', help='more request fields as JSON, e.g. \'{"top_p": 0.95}\'')
    r.add_argument('--timeout', type=int, default=900, help='seconds without a byte from the server')
    for p in (r, sub.add_parser('grade', help='grade a module'), sub.add_parser('selftest', help='check the grader')):
        p.add_argument('--sandbox', default='auto', choices=('auto', 'sandbox-exec', 'docker', 'none'))
        p.add_argument('--jobs', type=int, default=4, help='files graded at once')
    sub.choices['grade'].add_argument('code')
    sub.choices['grade'].add_argument('--json', help='also write the result here')
    t = sub.add_parser('table', help='a Markdown table of saved runs')
    t.add_argument('files', nargs='*')
    a = ap.parse_args()

    if a.cmd == 'table':
        return table(a.files or sorted(glob.glob(os.path.join(RESULTS, '*.json'))))
    if a.jobs < 1:
        sys.exit('--jobs must be 1 or more')
    if a.cmd == 'grade' and not os.path.isfile(a.code):
        sys.exit(f'no such file: {a.code}')
    if a.cmd == 'run':
        if a.turns < 0 or a.max_tokens < 0:
            sys.exit('--turns and --max-tokens must be 0 or more')
        if not a.base_url:
            sys.exit('give --base-url (or set OPENAI_BASE_URL), e.g. --base-url http://localhost:11434/v1 for Ollama')
        try:
            extra = json.loads(a.extra) if a.extra else {}
        except ValueError as e:
            sys.exit(f'--extra is not valid JSON: {e}')
        if not isinstance(extra, dict):
            sys.exit('--extra must be a JSON object, e.g. \'{"top_p": 0.95}\'')
    sandbox = pick_sandbox(a.sandbox)
    if sandbox == 'none':
        print('WARNING: --sandbox none runs the model\'s code on your machine with your rights.', file=sys.stderr)
    elif sandbox == 'sandbox-exec' and sys.version_info < (3, 12):
        print(f'note: the prompt asks for Python 3.12 and the code runs on this Python {sys.version.split()[0]}; '
              'code that needs 3.12 will fail here (Docker runs 3.12).', file=sys.stderr)
    if a.cmd == 'selftest':
        sys.exit(0 if selftest(sandbox, a.jobs) else 1)
    if a.cmd == 'grade':
        res = grade(a.code, jobs=a.jobs, sandbox=sandbox, echo=print)
        print(summary_line(res['summary']))
        if res['summary']['forbidden_words']:
            print('note: the code names', ', '.join(res['summary']['forbidden_words']))
        if a.json:
            with open(a.json, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=1, ensure_ascii=False)
        return
    if a.temperature is not None:
        extra['temperature'] = a.temperature
    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.join(RESULTS, re.sub(r'[^A-Za-z0-9._-]+', '_', a.model) + '-' +
                        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    def save(rec):
        with open(stem + '.json', 'w', encoding='utf-8') as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        for k, turn in enumerate([rec] + rec['feedback'], 1):
            if turn.get('code'):
                with open(f'{stem}.turn{k}.py', 'w', encoding='utf-8') as f:
                    f.write(turn['code'])

    try:
        rec = run_test(a.base_url, os.environ.get(a.api_key_env), a.model, max_tokens=a.max_tokens,
                       extra=extra or None, turns=a.turns, context=a.context, timeout=a.timeout, jobs=a.jobs,
                       sandbox=sandbox, save=save)
    except KeyboardInterrupt:
        sys.exit(f'\nstopped; the turns so far are in {os.path.relpath(stem, HERE)}.json')
    best = max(rec['final']['passed_by_turn'], default=0)
    print(f"\nThe Fax Test, {a.model}: {best}/40 files at best, CCITT "
          f"{max(rec['final']['ccitt_by_turn'], default=0)}/19; files by turn {rec['final']['passed_by_turn']}; "
          f"saved {os.path.relpath(stem, HERE)}.json")


if __name__ == '__main__':
    main()
