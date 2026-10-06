/*
 * Interactive terminal for the home page.
 * - Boots with an intro sequence
 * - Supports a small set of unix-like commands against a JSON site index
 * - Routes navigation through Jekyll URLs
 */
(function () {
  const body    = document.getElementById('term-body');
  const history = document.getElementById('term-history');
  const input   = document.getElementById('term-input');
  const caret   = document.getElementById('term-caret');
  const mirror  = document.getElementById('term-mirror');
  const cwdEl   = document.getElementById('term-cwd');
  if (!body || !input) return;

  // ---- Virtual filesystem & current directory ----------------------------
  // cwd is the path below home (~), e.g. [] = ~, ['reading'] = ~/reading,
  // ['reading','nerf'] = ~/reading/nerf. The tree is derived from INDEX in
  // listDir()/resolveDir() once the site index has loaded.
  let cwd = [];

  const homePath = () => '~' + (cwd.length ? '/' + cwd.join('/') : '');

  // Sync both the live input prompt and (re)used by writePrompt().
  const syncPrompt = () => { if (cwdEl) cwdEl.textContent = homePath(); };

  // Position the block caret right after the text the user has typed so it
  // blinks where they're editing, not at the far right of the line.
  const syncCaret = () => {
    if (!caret || !mirror) return;
    const pos = input.selectionStart == null ? input.value.length : input.selectionStart;
    // Mirror the text up to the cursor; a zero-width space keeps width stable.
    mirror.textContent = input.value.slice(0, pos) || '​';
    // subtract scrollLeft so the caret stays correct if the input overflows
    caret.style.left = (mirror.offsetWidth - input.scrollLeft) + 'px';
  };

  let INDEX = null;
  const cmdHistory = [];
  let cmdCursor = -1;

  // ------------------------------ helpers --------------------------------
  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));

  // Echo a command with the prompt that was active when it ran. Pass an
  // explicit `at` path for intro lines (which run before cwd moves).
  const writePrompt = (cmd, at) => {
    const line = document.createElement('p');
    line.className = 'term-line';
    const where = at == null ? homePath() : at;
    line.innerHTML = `<span class="term-prompt"><span class="term-cwd">${escapeHtml(where)}</span> $</span> ${escapeHtml(cmd)}`;
    history.appendChild(line);
  };

  const writeOut = (html, cls = '') => {
    const p = document.createElement('div');
    p.className = 'term-out ' + cls;
    p.innerHTML = html;
    history.appendChild(p);
  };

  const scrollDown = () => {
    body.scrollTop = body.scrollHeight;
    document.documentElement.scrollTop = Math.max(
      document.documentElement.scrollTop,
      0
    );
  };

  const clear = () => { history.innerHTML = ''; };

  // ------------------------------ jobs -----------------------------------
  // Animated / interactive commands (train, render, drive, ...) write into a
  // live block and run as a job. Ctrl+C / Esc (or q for interactive jobs)
  // stops it; an interactive job also gets first dibs on keystrokes.
  let job = null;

  const live = (cls = '') => {
    const el = document.createElement('div');
    el.className = 'term-out ' + cls;
    history.appendChild(el);
    return el;
  };

  // Run `step` every `ms` until it returns false or the job is interrupted.
  // opts: { onKey(e) -> handled?, interactive, onStop() }
  const runJob = (ms, step, opts = {}) => {
    const id = setInterval(() => {
      if (step() === false) finish();
      scrollDown();
    }, ms);
    const finish = () => {
      clearInterval(id);
      if (job && job.id === id) job = null;
    };
    job = { id, onKey: opts.onKey, interactive: !!opts.interactive, stop: () => { finish(); if (opts.onStop) opts.onStop(); } };
  };

  const interrupt = () => {
    if (!job) return false;
    const j = job;
    job = null;
    j.stop();
    writeOut('<span class="t-dim">^C</span>');
    scrollDown();
    return true;
  };

  const pick = (arr) => arr[(Math.random() * arr.length) | 0];

  // ------------------------------ filesystem -----------------------------
  // Top-level directories under ~ (cv is an external link, not a dir).
  const TOP_DIRS = ['about', 'now', 'reading', 'posts', 'contact'];

  // List the entries at a given path (array form). Returns
  // { dirs: [...], files: [{name, url, external}] } or null if the path is
  // not a real directory.
  function listDir(path) {
    if (!INDEX) return null;

    // ~  →  top-level dirs
    if (path.length === 0) {
      return { dirs: TOP_DIRS.filter((d) => d !== 'now' || (INDEX.now && INDEX.now.length)), files: [{ name: 'cv', url: INDEX.site.author.cv, external: true }] };
    }

    const head = path[0];

    if (head === 'reading') {
      const groups = INDEX.reading || [];
      // ~/reading  →  one dir per group id
      if (path.length === 1) {
        const ids = [];
        const seen = {};
        groups.forEach((r) => { if (!seen[r.group]) { seen[r.group] = 1; ids.push(r.group); } });
        return { dirs: ids, files: [] };
      }
      // ~/reading/<group>  →  the papers in that group as files
      if (path.length === 2) {
        const g = path[1];
        const items = groups.filter((r) => r.group === g);
        if (items.length === 0) return null;
        return { dirs: [], files: items.map((r) => ({ name: r.title, url: r.url, external: true })) };
      }
      return null;
    }

    if (head === 'posts') {
      if (path.length === 1) {
        return { dirs: [], files: (INDEX.posts || []).map((p) => ({ name: p.title, url: p.url })) };
      }
      return null;
    }

    // about / now / contact are leaf dirs whose `cat` text is their listing.
    if (path.length === 1 && (head === 'about' || head === 'now' || head === 'contact')) {
      return { dirs: [], files: [], leaf: head };
    }

    return null;
  }

  // Resolve a `cd` target (relative to cwd) into a new path array, or null
  // if it isn't a directory. Supports ., .., ~, /, and nested a/b segments.
  function resolveDir(arg) {
    let path;
    let rest;
    const a = arg.trim();

    if (a === '' || a === '~' || a === '/') return [];
    if (a.startsWith('~/')) { path = []; rest = a.slice(2); }
    else if (a.startsWith('/')) { path = []; rest = a.slice(1); }
    else { path = cwd.slice(); rest = a; }

    const segs = rest.split('/').filter((s) => s.length > 0);
    for (const seg of segs) {
      if (seg === '.') continue;
      if (seg === '..') { path.pop(); continue; }
      const here = listDir(path);
      if (!here || here.dirs.indexOf(seg) === -1) return null;
      path.push(seg);
    }
    return path;
  }

  // ------------------------------ commands -------------------------------
  const commands = {
    help: () => {
      const rows = [
        ['help',           'show this help'],
        ['whoami',         'who runs this place'],
        ['pwd',            'print working directory'],
        ['ls [dir]',       'list the current (or given) directory'],
        ['cd <dir>',       'change directory (try cd reading, cd .., cd ~)'],
        ['cat <file>',     'read a section: about, contact, now, affiliation'],
        ['find <pattern>', 'search posts + reading list (case-insensitive substring)'],
        ['git log',        'commit log of my research life'],
        ['open <name>',    'jump to a section in this browser tab'],
        ['theme [mode]',   'toggle, or set: theme dark | theme light'],
        ['clear',          'clear the terminal'],
        ['date',           'current server time'],
        ['train [epochs]', 'train a (very fake) model'],
        ['render',         'spin up a 3D gaussian donut'],
        ['drive',          'online mapping mini game (arrow keys)'],
        ['diffuse <text>', 'denoise your text from pure noise'],
        ['neofetch',       'system info, grad student edition'],
        ['ping sumin',     'check if I am awake'],
      ];
      const rendered = rows
        .map((r) => `  <span class="t-cmd">${r[0].padEnd(18)}</span><span class="t-desc">${r[1]}</span>`)
        .join('<br>');
      return rendered +
        `<br><br><span class="t-dim">tip: press <span class="t-cmd">Tab</span> to autocomplete commands and arguments, ` +
        `<span class="t-cmd">↑/↓</span> for history.</span>`;
    },

    whoami: () =>
      `<span class="t-strong">${INDEX.site.author.name}</span> · MS researcher · vision-centric autonomous driving<br>` +
      `<span class="t-dim">${INDEX.site.author.affiliation}</span>`,

    pwd: () => '/home/sumin1ee' + (cwd.length ? '/' + cwd.join('/') : ''),

    cd: (arg) => {
      const target = (arg || '').trim();
      const next = resolveDir(target);
      if (next === null) {
        return `<span class="t-err">cd: ${escapeHtml(target)}: No such directory</span>`;
      }
      // A leaf like ~/about has no children to enter — cat it instead.
      const here = listDir(next);
      if (here && here.leaf) {
        return `<span class="t-dim">${escapeHtml('~/' + next.join('/'))} is a leaf. try </span>` +
               `<span class="t-cmd">cat ${escapeHtml(here.leaf)}</span>`;
      }
      cwd = next;
      syncPrompt();
      return '';
    },

    ls: (arg) => {
      // ls with an argument lists that dir without moving into it.
      const path = arg && arg.trim() ? resolveDir(arg) : cwd;
      if (path === null) {
        return `<span class="t-err">ls: ${escapeHtml(arg.trim())}: No such directory</span>`;
      }
      const d = listDir(path);
      if (!d) return `<span class="t-err">ls: cannot access that path</span>`;
      if (d.leaf) {
        return `<span class="t-dim">(leaf: </span><span class="t-cmd">cat ${escapeHtml(d.leaf)}</span><span class="t-dim">)</span>`;
      }
      const dirs = d.dirs.map((n) => `<span class="t-dir">${escapeHtml(n)}/</span>`);
      const files = d.files.map((f) =>
        f.external
          ? `<a href="${f.url}" target="_blank" rel="noopener">${escapeHtml(f.name)}</a>`
          : `<a href="${f.url}">${escapeHtml(f.name)}</a>`
      );
      const all = dirs.concat(files);
      if (all.length === 0) return `<span class="t-dim">(empty)</span>`;
      // Papers/posts can be long titles — one per line; short dir lists inline.
      return d.files.length > 0 ? all.join('<br>') : all.join('  ');
    },

    cat: (arg) => {
      if (!arg) return `<span class="t-err">cat: missing operand</span>`;
      const a = arg.toLowerCase();
      switch (a) {
        case 'about':
        case 'about.md':
          return INDEX.site.tagline + '<br>' +
            `<span class="t-dim">A research notebook on Gaussian Splatting, Online Mapping, and end-to-end driving.</span>`;
        case 'contact':
        case 'contact.txt':
          return `email: <a href="mailto:${INDEX.site.author.email}">${INDEX.site.author.email}</a><br>` +
                 `github: <a href="https://github.com/${INDEX.site.author.github}" target="_blank">@${INDEX.site.author.github}</a><br>` +
                 `cv: <a href="${INDEX.site.author.cv}" target="_blank">${INDEX.site.author.cv}</a>`;
        case 'now':
        case 'now.md':
          if (!INDEX.now || INDEX.now.length === 0) return '<span class="t-dim">(nothing right now)</span>';
          return INDEX.now.map((n) => `› ${stripMd(n)}`).join('<br>');
        case 'affiliation':
          return `<a href="${INDEX.site.author.affiliation_url}" target="_blank">${INDEX.site.author.affiliation}</a>`;
        default:
          return `<span class="t-err">cat: ${escapeHtml(arg)}: No such file</span>`;
      }
    },

    find: (arg) => {
      if (!arg) return `<span class="t-err">find: usage: find &lt;pattern&gt;</span>`;
      const q = arg.toLowerCase();
      const hits = [];

      INDEX.reading.forEach((r) => {
        const hay = `${r.title} ${r.authors} ${r.venue} ${r.note} ${r.group_label}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            kind: 'paper',
            line: `<span class="t-dir">reading/${r.group}/</span> <a href="${r.url}" target="_blank">${escapeHtml(r.title)}</a> <span class="t-dim">${r.venue} ${r.year}</span>`
          });
        }
      });

      INDEX.posts.forEach((p) => {
        const hay = `${p.title} ${(p.tags || []).join(' ')}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            kind: 'post',
            line: `<span class="t-dir">posts/</span> <a href="${p.url}">${escapeHtml(p.title)}</a>`
          });
        }
      });

      INDEX.now.forEach((n) => {
        if (n.toLowerCase().includes(q)) {
          hits.push({
            kind: 'now',
            line: `<span class="t-dir">now/</span> ${stripMd(n)}`
          });
        }
      });

      if (hits.length === 0) {
        return `<span class="t-dim">no matches for "${escapeHtml(arg)}"</span>`;
      }

      const header = `<span class="t-dim">${hits.length} ${hits.length === 1 ? 'match' : 'matches'}</span><br>`;
      return header + hits.slice(0, 40).map((h) => h.line).join('<br>') +
             (hits.length > 40 ? `<br><span class="t-dim">… ${hits.length - 40} more (refine your query)</span>` : '');
    },

    open: (arg) => {
      if (!arg) return `<span class="t-err">open: missing operand</span>`;
      const a = arg.toLowerCase();
      const routes = {
        posts:    '/posts/',
        reading:  '/reading/',
        cv:       INDEX.site.author.cv,
        github:   'https://github.com/' + INDEX.site.author.github,
        home:     '/',
        '/':      '/',
      };
      const url = routes[a] || (a.startsWith('http') ? a : null);
      if (!url) return `<span class="t-err">open: unknown target: ${escapeHtml(arg)}</span>`;
      writeOut(`<span class="t-dim">opening ${escapeHtml(url)} …</span>`);
      setTimeout(() => {
        if (url.startsWith('http')) window.open(url, '_blank');
        else window.location.href = url;
      }, 250);
      return '';
    },

    theme: (arg) => {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') || 'dark';
      const want = (arg || '').trim().toLowerCase();

      // `theme` with no argument toggles; `theme dark|light` sets explicitly.
      let next;
      if (!want) {
        next = cur === 'dark' ? 'light' : 'dark';
      } else if (want === 'dark' || want === 'light') {
        if (want === cur) {
          return `<span class="t-dim">already in ${cur} mode.</span>`;
        }
        next = want;
      } else {
        return `<span class="t-err">theme: '${escapeHtml(want)}' is not a valid mode.</span> ` +
               `Try <span class="t-cmd">theme dark</span>, <span class="t-cmd">theme light</span>, or just <span class="t-cmd">theme</span> to toggle.`;
      }

      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme-v2', next); } catch (e) {}
      return `theme → ${next}`;
    },

    clear: () => { clear(); return ''; },

    date: () => new Date().toString(),

    git: (arg) => {
      const sub = (arg || '').trim().split(/\s+/)[0].toLowerCase();
      if (!sub) return `<span class="t-dim">usage: git &lt;command&gt;. try <span class="t-cmd">git log</span></span>`;
      if (sub === 'log') return gitLog();
      if (sub === 'status') {
        return `On branch <span class="t-cmd">main</span><br>` +
               `Your research is ahead of 'origin/main' by 2 papers.<br>` +
               `<span class="t-dim">  (use "git push" to publish)</span><br><br>` +
               `Changes not staged for commit:<br>` +
               `  <span class="t-err">modified:</span>   under_review.tex<br>` +
               `  <span class="t-err">modified:</span>   ReSMap.tex<br>` +
               `  <span class="t-err">untracked:</span>  next-idea.md`;
      }
      if (sub === 'blame') return `<span class="t-dim">blame: don't blame me, blame the reviewers.</span>`;
      if (sub === 'push')  return `<span class="t-dim">remote: hold on, submission window opens in T-7 months.</span>`;
      if (sub === 'pull')  return `<span class="t-dim">Already up to date with reality.</span>`;
      if (sub === 'commit') return `<span class="t-err">git: please use real git to commit research progress.</span>`;
      return `<span class="t-err">git: '${escapeHtml(sub)}' is not a supported command here. Try <span class="t-cmd">git log</span>.</span>`;
    },

    // ---- ML toys ----------------------------------------------------------

    // Fake training run: tqdm-ish bar, per-epoch log, ASCII loss curve.
    // Occasionally dies of CUDA OOM, as is tradition.
    train: (arg) => {
      const E = Math.min(30, Math.max(1, parseInt(arg, 10) || 8));
      const oomAt = Math.random() < 0.18 ? 1 + ((Math.random() * E) | 0) : -1;
      writeOut(
        `<span class="t-dim">$ python tools/train.py configs/toy_bev.py --epochs ${E}</span><br>` +
        `loading nuScenes v1.0-trainval · 28,130 samples · 6 cams<br>` +
        `model: TinyBEV-R50 · 41.2M params · 8 × GPU · batch 4<br>` +
        `<span class="t-dim">(Ctrl+C to stop)</span>`
      );
      const bar = live('t-pre');
      const log = live('t-pre');
      const STEPS = 20, hist = [];
      let ep = 1, st = 0, map = 0, loss = 2.6 + Math.random() * 0.3;
      runJob(45, () => {
        st++;
        loss = Math.max(0.06, loss * (0.985 - Math.random() * 0.01) + (Math.random() - 0.5) * 0.03);
        if (ep === oomAt && st === 13) {
          bar.innerHTML = '';
          log.innerHTML += `<span class="t-err">RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB ` +
            `(GPU 0; 23.65 GiB total capacity; 21.93 GiB already allocated)</span>\n` +
            `<span class="t-dim">tip: batch_size=1 and a small prayer 🙏  (run train again)</span>`;
          return false;
        }
        const w = 24, fill = Math.round((st / STEPS) * w);
        bar.innerHTML = `epoch ${String(ep).padStart(2)}/${E} <span class="t-cmd">${'█'.repeat(fill)}</span>` +
          `${'░'.repeat(w - fill)} ${String(Math.round((st / STEPS) * 100)).padStart(3)}%  ` +
          `loss ${loss.toFixed(4)}  ${(4.6 + Math.random()).toFixed(1)}it/s`;
        if (st < STEPS) return;
        map = 30 + 32 * (1 - Math.exp(-ep / 3.5)) + Math.random();
        hist.push(loss);
        log.innerHTML += `epoch ${String(ep).padStart(2)}/${E}  loss ${loss.toFixed(4)}  val mAP ${map.toFixed(1)}\n`;
        st = 0; ep++;
        if (ep <= E) return;
        bar.innerHTML = '';
        log.innerHTML += '\n' + lossPlot(hist) +
          `\n\n<span class="t-cmd">✓</span> saved ckpt/epoch_${E}.pth · val mAP ${map.toFixed(1)} ` +
          `<span class="t-dim">(reviewer 2 wants more ablations)</span>`;
        return false;
      });
      return '';
    },

    // A spinning torus made of 3D gaussians, rasterized to ASCII (donut.c tribute).
    render: () => {
      const W = 58, H = 22, N = 4200, SH = '.,-~:;=!*#$@';
      const g = [];
      for (let i = 0; i < N; i++) {
        const u = Math.random() * Math.PI * 2, v = Math.random() * Math.PI * 2;
        const R = 2, r = 0.95, cv = Math.cos(v);
        g.push([(R + r * cv) * Math.cos(u), r * Math.sin(v), (R + r * cv) * Math.sin(u),
                cv * Math.cos(u), Math.sin(v), cv * Math.sin(u)]);
      }
      const hud = live();
      const el = live('t-pre t-render');
      let A = 0.9, B = 0, f = 0, last = performance.now(), fps = 0;
      runJob(55, () => {
        const zb = new Float32Array(W * H), lum = new Int8Array(W * H).fill(-1);
        const cA = Math.cos(A), sA = Math.sin(A), cB = Math.cos(B), sB = Math.sin(B);
        for (const [x, y, z, nx, ny, nz] of g) {
          const y1 = y * cA - z * sA, z1 = y * sA + z * cA;
          const x2 = x * cB + z1 * sB, z2 = -x * sB + z1 * cB;
          const ny1 = ny * cA - nz * sA, nz1 = ny * sA + nz * cA;
          const nz2 = -nx * sB + nz1 * cB;
          const D = 1 / (z2 + 6);
          const px = Math.round(W / 2 + x2 * D * W * 0.95), py = Math.round(H / 2 - y1 * D * H * 0.95);
          if (px < 0 || px >= W || py < 0 || py >= H) continue;
          const k = py * W + px;
          if (D > zb[k]) {
            zb[k] = D;
            const L = (ny1 - nz2) * 0.7071; // light from above, towards the viewer
            lum[k] = Math.max(0, Math.min(SH.length - 1, Math.round(L * (SH.length - 1))));
          }
        }
        let out = '';
        for (let r = 0; r < H; r++) {
          for (let c = 0; c < W; c++) { const l = lum[r * W + c]; out += l < 0 ? ' ' : SH[l]; }
          out += '\n';
        }
        el.textContent = out;
        const now = performance.now();
        fps = 0.9 * fps + 0.1 * (1000 / (now - last)); last = now;
        hud.innerHTML = `<span class="t-dim">3DGS viewer · ${N.toLocaleString()} gaussians · ${fps.toFixed(0)} fps · Esc to stop</span>`;
        A += 0.07; B += 0.035;
        return ++f < 360;
      });
      return '';
    },

    // Online-mapping mini game: steer the ego car; the map only exists
    // where you've already "perceived" it. Everything else is fog.
    drive: () => {
      const W = 33, H = 17, CAR = H - 3, RANGE = 8;
      const cxAt = (s) => Math.round(W / 2 + 5 * Math.sin(s / 9) + 2 * Math.sin(s / 4.3));
      const cars = [];
      for (let k = 16; k < 6000; k += 5 + ((Math.random() * 7) | 0)) cars.push({ s: k, lane: ((Math.random() * 3) | 0) - 1 });
      let s = 0, x = cxAt(0), speed = 1, acc = 0, dead = null;

      writeOut(`<span class="t-dim">online mapping sim · perception range ${RANGE * 2} m · fog = not mapped yet<br>` +
               `← → steer · ↑ ↓ speed · q quit</span>`);
      const el = live('t-pre t-drive');

      const draw = () => {
        const rows = [];
        for (let r = 0; r < H; r++) {
          const ahead = CAR - r, ws = s + ahead, c = cxAt(ws);
          let line = '';
          for (let col = 0; col < W; col++) {
            const d = col - c;
            if (r === CAR && col === x) { line += `<span class="t-cmd t-strong">${dead ? 'X' : 'A'}</span>`; continue; }
            if (ahead > RANGE) { line += (col + r) % 3 === 0 ? '<span class="t-fog">.</span>' : ' '; continue; }
            const car = cars.find((o) => o.s === ws && cxAt(o.s) + o.lane * 4 === col);
            if (car) { line += '<span class="t-err">#</span>'; continue; }
            // freshly perceived rows at the edge of range flicker like new predictions
            const fresh = ahead >= RANGE - 1 && Math.random() < 0.35;
            if (d === -6 || d === 6) line += fresh ? '<span class="t-dim">!</span>' : '|';
            else if ((d === -2 || d === 2) && ws % 2 === 0) line += `<span class="t-cmd">${fresh ? '.' : ':'}</span>`;
            else line += ' ';
          }
          rows.push(line);
        }
        const meters = s * 2;
        rows.push('');
        rows.push(`speed ${'▮'.repeat(speed)}${'▯'.repeat(3 - speed)}  ${String(meters).padStart(4)} m  ` +
                  `map ${s + RANGE} rows · ${Math.floor((s + RANGE) / 2) * 3 + 2} inst`);
        el.innerHTML = rows.join('\n');
      };

      runJob(90, () => {
        if (dead) return false;
        acc += speed * 0.5;
        while (acc >= 1) {
          acc -= 1; s++;
          if (Math.abs(x - cxAt(s)) >= 6) dead = 'offroad';
          else if (cars.some((o) => o.s === s && cxAt(o.s) + o.lane * 4 === x)) dead = 'crash';
          if (dead) break;
        }
        draw();
        if (!dead) return;
        writeOut(dead === 'crash'
          ? `💥 crash at ${s * 2} m. the planner has notes. <span class="t-dim">(drive to retry)</span>`
          : `🌾 off-road at ${s * 2} m. should have trusted the centerline. <span class="t-dim">(drive to retry)</span>`);
        return false;
      }, {
        interactive: true,
        onKey: (e) => {
          if (e.key === 'ArrowLeft') x--;
          else if (e.key === 'ArrowRight') x++;
          else if (e.key === 'ArrowUp') speed = Math.min(3, speed + 1);
          else if (e.key === 'ArrowDown') speed = Math.max(0, speed - 1);
          else return false;
          draw();
          return true;
        },
      });
      draw();
      return '';
    },

    // Denoise any text, DDPM-style.
    diffuse: (arg) => {
      const text = (arg || 'hello, world').slice(0, 80);
      const T = 24, NOISE = '░▒▓█#%&@$*+=?/<>~^01';
      const at = [...text].map((ch) => (ch === ' ' ? -1 : Math.random() * 0.85));
      const el = live('t-pre');
      let f = 0;
      runJob(55, () => {
        const p = f / T;
        if (f++ >= T) {
          el.innerHTML = `<span class="t-dim">t=   0</span>  <span class="t-cmd">${escapeHtml(text)}</span>  <span class="t-dim">(${T} steps)</span>`;
          return false;
        }
        const s = [...text].map((ch, i) =>
          at[i] < 0 || p >= at[i] + 0.15 ? escapeHtml(ch) : `<span class="t-dim">${escapeHtml(pick(NOISE))}</span>`
        ).join('');
        el.innerHTML = `<span class="t-dim">t=${String(Math.round((1 - p) * 1000)).padStart(4)}</span>  ${s}`;
      });
      return '';
    },

    neofetch: () => {
      const art = gaussArt();
      const a = INDEX.site.author || {};
      const years = (Date.now() - new Date('2019-03-04').getTime()) / (365.25 * 864e5);
      const papers = (INDEX.reading || []).length, posts = (INDEX.posts || []).length;
      const theme = document.documentElement.getAttribute('data-theme') || 'dark';
      const kv = (k, v) => `<span class="t-cmd">${k.padEnd(9)}</span>${v}`;
      const info = [
        `<span class="t-cmd">sumin</span>@<span class="t-cmd">ircv-lab</span>`,
        '─'.repeat(22),
        kv('os', 'Hanyang Univ. · MS, Automotive Eng.'),
        kv('host', 'IRCV Lab'),
        kv('kernel', 'pytorch + cuda, fueled by ☕'),
        kv('uptime', `${Math.floor(years)} yrs ${Math.floor((years % 1) * 12)} mos (since 2019-03)`),
        kv('packages', `${posts} posts · ${papers} papers on the shelf`),
        kv('shell', 'zsh (you are in it)'),
        kv('research', '3DGS · online HD maps · e2e driving'),
        kv('gpu', '1× very tired GPU'),
        kv('theme', theme),
        '',
        ['--accent', '--accent-warn', '--ink-0', '--ink-2', '--ink-3', '--border-strong']
          .map((v) => `<span class="nf-sw" style="background:var(${v})"></span>`).join(''),
      ];
      const n = Math.max(art.length, info.length);
      const lines = [];
      for (let i = 0; i < n; i++) {
        lines.push(`<span class="t-cmd">${escapeHtml(art[i] || ' '.repeat(art[0].length))}</span>   ${info[i] || ''}`);
      }
      return `<div class="t-pre">${lines.join('\n')}</div>`;
    },

    vim: () => {
      vimMode = true;
      return Array(5).fill('<span class="t-cmd">~</span>').join('<br>') +
        `<br><span class="t-dim">"untitled.tex" [New File]</span>` +
        `<br><span class="t-dim">you are now in vim. good luck getting out.</span>`;
    },

    ping: (arg) => {
      const host = (arg || '').trim().toLowerCase();
      if (!host) return `<span class="t-dim">usage: ping &lt;host&gt;. try <span class="t-cmd">ping sumin</span></span>`;
      if (!['sumin', 'sumin1ee', 'sumin-lee'].includes(host)) {
        return `<span class="t-err">ping: ${escapeHtml(host)}: Name or service not known.</span> ` +
               `<span class="t-dim">try <span class="t-cmd">ping sumin</span></span>`;
      }
      const email = (INDEX.site.author || {}).email || '';
      const REASONS = ['probably asleep', 'training a model', 'in a lab meeting', 'reading arXiv',
                       'waiting for a GPU', 'writing a rebuttal', 'debugging a NaN loss'];
      const el = live();
      el.innerHTML = 'PING sumin (hanyang.ac.kr) 56(84) bytes of data.';
      const times = [];
      runJob(650, () => {
        if (times.length < 4) {
          const t = +(0.5 + Math.random() * 5).toFixed(1);
          times.push(t);
          el.innerHTML += `<br>64 bytes from sumin: icmp_seq=${times.length} ttl=64 time=${t} hrs ` +
                          `<span class="t-dim">(${pick(REASONS)})</span>`;
          return;
        }
        const avg = (times.reduce((x, y) => x + y, 0) / times.length).toFixed(1);
        el.innerHTML += `<br><br>--- sumin ping statistics ---<br>` +
          `4 packets transmitted, 4 received, 0% packet loss<br>` +
          `rtt min/avg/max = ${Math.min(...times)}/${avg}/${Math.max(...times)} hrs<br>` +
          (email ? `<span class="t-dim">lower latency: </span><a href="mailto:${escapeHtml(email)}">${escapeHtml(email)}</a>` : '');
        return false;
      });
      return '';
    },

    // easter eggs
    sudo: () => `<span class="t-err">sudo: permission denied. this is a static site, friend.</span>`,
    rm:   () => `<span class="t-err">rm: nice try.</span>`,
    exit: () => `<span class="t-dim">(close the tab to exit)</span>`,
    hello: () => `hi 👋`,
    coffee: () => `☕`,
  };

  let vimMode = false;

  // Loss-per-epoch as a small ASCII scatter plot.
  function lossPlot(h) {
    const H = 6, rep = Math.max(1, Math.floor(36 / h.length));
    const pts = h.flatMap((v) => Array(rep).fill(v));
    const lo = Math.min(...h), hi = Math.max(...h);
    const rows = [];
    for (let r = H - 1; r >= 0; r--) {
      const label = r === H - 1 ? hi.toFixed(2) : r === 0 ? lo.toFixed(2) : '';
      let line = label.padStart(5) + ' |';
      for (const v of pts) {
        const y = hi === lo ? 0 : Math.round(((v - lo) / (hi - lo)) * (H - 1));
        line += y === r ? '<span class="t-cmd">*</span>' : ' ';
      }
      rows.push(line);
    }
    rows.push('      +' + '-'.repeat(pts.length) + ' epoch');
    return '<span class="t-dim">train/loss</span>\n' + rows.join('\n');
  }

  // A rotated anisotropic 2D gaussian, drawn in ASCII density characters.
  function gaussArt() {
    const H = 12, W = 26, th = 0.5, c = Math.cos(th), s = Math.sin(th), SH = ' .:-=+*#%@';
    const out = [];
    for (let r = 0; r < H; r++) {
      let line = '';
      for (let q = 0; q < W; q++) {
        const x = ((q - W / 2 + 0.5) / W) * 2.2, y = ((r - H / 2 + 0.5) / H) * 2.2;
        const u = x * c + y * s, v = -x * s + y * c;
        const g = Math.exp(-((u * u) / (2 * 0.45 ** 2) + (v * v) / (2 * 0.2 ** 2)));
        line += SH[Math.min(9, Math.round(g * 9))];
      }
      out.push(line);
    }
    return out;
  }

  function stripMd(s) {
    return escapeHtml(s)
      .replace(/\*\*([^*]+)\*\*/g, '<span class="t-strong">$1</span>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  // Render a `git log --decorate --oneline`-ish view.
  function gitLog() {
    const commits = (INDEX.git_log || []);
    if (commits.length === 0) {
      return `<span class="t-dim">(no commits yet)</span>`;
    }
    return commits.map(formatCommit).join('<br>');
  }

  function formatCommit(c) {
    const hash = `<span class="git-hash">${escapeHtml(c.hash)}</span>`;
    const author = `<span class="git-author">${escapeHtml(c.author)}</span>`;
    const date = `<span class="git-date">${escapeHtml(humanDate(c.date))}</span>`;
    const refs = (c.refs && c.refs.length > 0)
      ? ' <span class="git-refs">(' +
          c.refs.map(r => {
            if (r === 'HEAD') return '<span class="git-head">HEAD</span>';
            if (r.startsWith('tag:')) return `<span class="git-tag">${escapeHtml(r)}</span>`;
            if (r === 'main' || r === 'master') return `<span class="git-branch">${escapeHtml(r)}</span>`;
            return `<span class="git-ref">${escapeHtml(r)}</span>`;
          }).join(', ') +
        ')</span>'
      : '';
    const msg = `<span class="git-msg">${escapeHtml(c.msg)}</span>`;
    return `${hash} ${date} ${author}${refs}<br>&nbsp;&nbsp;&nbsp;&nbsp;${msg}`;
  }

  function humanDate(iso) {
    // ISO date -> "Mon May 12 2026" style, locale-independent
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    return `${days[d.getUTCDay()]} ${months[d.getUTCMonth()]} ${String(d.getUTCDate()).padStart(2,'0')} ${d.getUTCFullYear()}`;
  }

  // ---------------------------- completion -------------------------------
  // Candidate sets for the second word of each command, mirroring the
  // switch/route tables above. `git` takes subcommands instead of files.
  const COMMAND_NAMES = () => Object.keys(commands).concat(['git']);

  const ARG_CANDIDATES = {
    cat:   ['about', 'now', 'contact', 'affiliation'],
    open:  ['posts', 'reading', 'cv', 'github', 'home'],
    git:   ['log', 'status', 'blame', 'push', 'pull', 'commit'],
    theme: ['dark', 'light'],
    ping:  ['sumin'],
  };

  // Longest common prefix of a list of strings.
  function commonPrefix(list) {
    if (list.length === 0) return '';
    let p = list[0];
    for (let i = 1; i < list.length; i++) {
      while (!list[i].startsWith(p)) p = p.slice(0, -1);
      if (!p) break;
    }
    return p;
  }

  // Given the current input, return { candidates, replaceFrom } where
  // replaceFrom is the index in the string where the completed token starts.
  function getCompletion(value) {
    // Leading whitespace is preserved; we only complete the last token.
    const m = value.match(/^(\s*)(.*)$/);
    const lead = m[1];
    const rest = m[2];
    const parts = rest.split(/\s+/);
    const onFirstWord = parts.length === 1 && !/\s$/.test(value);

    if (onFirstWord) {
      const frag = parts[0].toLowerCase();
      const all = Array.from(new Set(COMMAND_NAMES())).sort();
      // On an empty line, only advertise the documented commands (skip the
      // easter eggs); when there's a prefix, complete against everything.
      const documented = ['help', 'whoami', 'pwd', 'ls', 'cd', 'cat', 'find', 'git', 'open', 'theme', 'clear', 'date',
                          'train', 'render', 'drive', 'diffuse', 'neofetch', 'ping'];
      const pool = frag === '' ? documented.slice().sort() : all;
      const cands = frag === '' ? pool : pool.filter((n) => n.startsWith(frag));
      return { candidates: cands, replaceFrom: lead.length, isCommand: true };
    }

    // Completing an argument. The command is parts[0]; the fragment is the
    // last token (which is '' if the input ends with a space).
    const name = parts[0].toLowerCase();
    const endsWithSpace = /\s$/.test(value);
    const frag = endsWithSpace ? '' : parts[parts.length - 1].toLowerCase();

    // cd / ls complete against the directories under the current dir (or under
    // the parent of a partially-typed `a/b` path).
    let pool;
    if (name === 'cd' || name === 'ls') {
      const slash = frag.lastIndexOf('/');
      const dirPrefix = slash === -1 ? '' : frag.slice(0, slash + 1); // keep trailing /
      const base = slash === -1 ? frag : frag.slice(slash + 1);
      const at = dirPrefix ? resolveDir(dirPrefix) : cwd;
      const here = at ? listDir(at) : null;
      const dirs = here ? here.dirs.slice() : [];
      const matched = base === '' ? dirs : dirs.filter((d) => d.toLowerCase().startsWith(base));
      // Re-attach the dirPrefix so completion fills the full token.
      const cands = matched.map((d) => dirPrefix + d).sort();
      const replaceFrom = endsWithSpace ? value.length : value.length - frag.length;
      return { candidates: cands, replaceFrom };
    }

    pool = ARG_CANDIDATES[name];
    if (!pool) return { candidates: [], replaceFrom: value.length };

    const cands = frag === '' ? pool.slice() : pool.filter((c) => c.startsWith(frag));
    // Where the fragment begins in the original string.
    const replaceFrom = endsWithSpace ? value.length : value.length - frag.length;
    return { candidates: cands.sort(), replaceFrom };
  }

  function handleTab() {
    const value = input.value;
    const { candidates, replaceFrom } = getCompletion(value);
    if (candidates.length === 0) return;

    const head = value.slice(0, replaceFrom);
    const completed = commonPrefix(candidates);

    if (candidates.length === 1) {
      // Unique match: fill it in and add a trailing space so the next Tab
      // moves on to argument completion.
      input.value = head + candidates[0] + ' ';
    } else {
      // Multiple matches: extend to the longest common prefix, and if that
      // adds nothing new, list the options (bash-style).
      const current = value.slice(replaceFrom);
      if (completed.length > current.length) {
        input.value = head + completed;
      } else {
        writePrompt(value);
        writeOut(candidates.map((c) => `<span class="t-dir">${escapeHtml(c)}</span>`).join('  '));
        scrollDown();
      }
    }
    syncCaret();
  }

  // ------------------------------ execute --------------------------------
  function exec(raw) {
    const cmd = raw.trim();
    if (!cmd) return;
    if (job) interrupt();
    writePrompt(cmd);
    if (vimMode) {
      if (/^:(q|q!|qa!|wq|x)$/.test(cmd)) {
        vimMode = false;
        writeOut('you escaped vim 🎉 <span class="t-dim">(+10 xp. most people never do)</span>');
      } else {
        writeOut(`<span class="t-err">E492: Not an editor command: ${escapeHtml(cmd)}</span> <span class="t-dim">(hint: :q!)</span>`);
      }
      scrollDown();
      return;
    }
    const parts = cmd.split(/\s+/);
    const name = parts[0].toLowerCase();
    const arg  = parts.slice(1).join(' ');
    const fn = commands[name];
    if (!fn) {
      writeOut(`<span class="t-err">${escapeHtml(name)}: command not found.</span> Type <span class="t-cmd">help</span>.`);
    } else {
      const out = fn(arg);
      if (out) writeOut(out);
    }
    scrollDown();
  }

  // ------------------------------ intro ----------------------------------
  function bootIntro() {
    const lines = [
      { type: 'cmd', text: 'whoami' },
      { type: 'out', html: commands.whoami() },
      { type: 'cmd', text: 'cat ./affiliation' },
      { type: 'out', html: commands.cat('affiliation') },
      { type: 'cmd', text: 'ls' },
      { type: 'out', html: commands.ls() },
      { type: 'tip', text: 'type <span class="t-cmd">help</span>, then try <span class="t-cmd">cd reading</span>, <span class="t-cmd">ls</span>, or <span class="t-cmd">find gaussian</span>.' },
    ];
    let i = 0;
    const tick = () => {
      if (i >= lines.length) return;
      const l = lines[i++];
      if (l.type === 'cmd') writePrompt(l.text);
      else if (l.type === 'out') writeOut(l.html);
      else if (l.type === 'tip') writeOut(`<span class="t-dim">› ${l.text}</span>`, 'term-tip');
      scrollDown();
      setTimeout(tick, 220);
    };
    tick();
  }

  // ------------------------------ wiring ---------------------------------
  syncPrompt();
  fetch('/assets/data/index.json')
    .then((r) => r.json())
    .then((data) => {
      INDEX = data;
      bootIntro();
    })
    .catch(() => {
      INDEX = { site: { author: {}, title: '', tagline: '' }, posts: [], reading: [], now: [] };
      writeOut('<span class="t-err">terminal: failed to load site index.</span>');
    });

  input.addEventListener('keydown', (e) => {
    const ctrlC = e.key === 'c' && (e.ctrlKey || e.metaKey) && !window.getSelection().toString();
    if (job) {
      if (ctrlC || e.key === 'Escape' || (job.interactive && e.key === 'q')) { e.preventDefault(); interrupt(); return; }
      if (job.onKey && job.onKey(e)) { e.preventDefault(); return; }
      if (job.interactive) { e.preventDefault(); return; } // the game owns the keyboard
    } else if (vimMode && ctrlC) {
      e.preventDefault();
      writeOut('<span class="t-dim">Type  :qa!  and press &lt;Enter&gt; to abandon all changes and exit Vim</span>');
      scrollDown();
      return;
    }
    if (e.key === 'Enter') {
      const v = input.value;
      input.value = '';
      if (v.trim()) {
        cmdHistory.push(v);
        cmdCursor = cmdHistory.length;
      }
      exec(v);
    } else if (e.key === 'Tab') {
      e.preventDefault();
      handleTab();
    } else if (e.key === 'ArrowUp') {
      if (cmdHistory.length === 0) return;
      cmdCursor = Math.max(0, cmdCursor - 1);
      input.value = cmdHistory[cmdCursor] || '';
      e.preventDefault();
    } else if (e.key === 'ArrowDown') {
      cmdCursor = Math.min(cmdHistory.length, cmdCursor + 1);
      input.value = cmdHistory[cmdCursor] || '';
      e.preventDefault();
    } else if (e.key === 'l' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      clear();
    }
    // run after the browser applies the keystroke so caret tracks the result
    requestAnimationFrame(syncCaret);
  });

  // keep the caret glued to the text/cursor through every kind of edit
  ['input', 'click', 'keyup', 'focus', 'select'].forEach((ev) =>
    input.addEventListener(ev, syncCaret)
  );

  // Blink the caret only while the input is focused. Toggling a class is more
  // robust than a CSS sibling selector (which breaks if the DOM order shifts).
  const setBlink = (on) => { if (caret) caret.classList.toggle('is-blinking', on); };
  input.addEventListener('focus', () => setBlink(true));
  input.addEventListener('blur',  () => setBlink(false));
  setBlink(document.activeElement === input);

  // click anywhere on the terminal -> focus input
  document.getElementById('terminal').addEventListener('click', (e) => {
    if (window.getSelection().toString().length === 0) input.focus();
  });
  // auto-focus on load (but not aggressively if user is scrolling)
  setTimeout(() => { input.focus(); syncCaret(); setBlink(document.activeElement === input); }, 600);
})();
