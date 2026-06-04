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
  if (!body || !input) return;

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

  const writePrompt = (cmd) => {
    const line = document.createElement('p');
    line.className = 'term-line';
    line.innerHTML = `<span class="term-prompt">$</span> ${escapeHtml(cmd)}`;
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

  // ------------------------------ commands -------------------------------
  const commands = {
    help: () => {
      const rows = [
        ['help',           'show this help'],
        ['whoami',         'who runs this place'],
        ['pwd',            'print working directory'],
        ['ls',             'list site sections'],
        ['cat <file>',     'read a section: about, contact, now, affiliation'],
        ['find <pattern>', 'search posts + reading list (case-insensitive substring)'],
        ['git log',        'commit log of my research life'],
        ['open <name>',    'jump to a section in this browser tab'],
        ['theme',          'toggle dark / light mode'],
        ['clear',          'clear the terminal'],
        ['date',           'current server time'],
      ];
      const rendered = rows
        .map((r) => `  <span class="t-cmd">${r[0].padEnd(18)}</span><span class="t-desc">${r[1]}</span>`)
        .join('<br>');
      return rendered +
        `<br><br><span class="t-dim">tip: press <span class="t-cmd">Tab</span> to autocomplete commands and arguments, ` +
        `<span class="t-cmd">↑/↓</span> for history.</span>`;
    },

    whoami: () =>
      `<span class="t-strong">${INDEX.site.author.name}</span> — MS researcher · vision-centric autonomous driving<br>` +
      `<span class="t-dim">${INDEX.site.author.affiliation}</span>`,

    pwd: () => '/home/sumin1ee',

    ls: () => {
      const items = ['about', 'now', 'reading', 'posts', 'contact', 'cv'];
      return items.map((i) => `<span class="t-dir">${i}/</span>`).join('  ');
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

    theme: () => {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      return `theme → ${next}`;
    },

    clear: () => { clear(); return ''; },

    date: () => new Date().toString(),

    git: (arg) => {
      const sub = (arg || '').trim().split(/\s+/)[0].toLowerCase();
      if (!sub) return `<span class="t-dim">usage: git &lt;command&gt; — try <span class="t-cmd">git log</span></span>`;
      if (sub === 'log') return gitLog();
      if (sub === 'status') {
        return `On branch <span class="t-cmd">main</span><br>` +
               `Your research is ahead of 'origin/main' by 2 papers.<br>` +
               `<span class="t-dim">  (use "git push" to publish)</span><br><br>` +
               `Changes not staged for commit:<br>` +
               `  <span class="t-err">modified:</span>   CamoSplat.tex<br>` +
               `  <span class="t-err">modified:</span>   ReSMap.tex<br>` +
               `  <span class="t-err">untracked:</span>  next-idea.md`;
      }
      if (sub === 'blame') return `<span class="t-dim">blame: don't blame me, blame the reviewers.</span>`;
      if (sub === 'push')  return `<span class="t-dim">remote: hold on, submission window opens in T-7 months.</span>`;
      if (sub === 'pull')  return `<span class="t-dim">Already up to date with reality.</span>`;
      if (sub === 'commit') return `<span class="t-err">git: please use real git to commit research progress.</span>`;
      return `<span class="t-err">git: '${escapeHtml(sub)}' is not a supported command here. Try <span class="t-cmd">git log</span>.</span>`;
    },

    // easter eggs
    sudo: () => `<span class="t-err">sudo: permission denied — this is a static site, friend.</span>`,
    rm:   () => `<span class="t-err">rm: nice try.</span>`,
    exit: () => `<span class="t-dim">(close the tab to exit)</span>`,
    hello: () => `hi 👋`,
    coffee: () => `☕`,
  };

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
    cat:  ['about', 'now', 'contact', 'affiliation'],
    open: ['posts', 'reading', 'cv', 'github', 'home'],
    git:  ['log', 'status', 'blame', 'push', 'pull', 'commit'],
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
      const documented = ['help', 'whoami', 'pwd', 'ls', 'cat', 'find', 'git', 'open', 'theme', 'clear', 'date'];
      const pool = frag === '' ? documented.slice().sort() : all;
      const cands = frag === '' ? pool : pool.filter((n) => n.startsWith(frag));
      return { candidates: cands, replaceFrom: lead.length, isCommand: true };
    }

    // Completing an argument. The command is parts[0]; the fragment is the
    // last token (which is '' if the input ends with a space).
    const name = parts[0].toLowerCase();
    const pool = ARG_CANDIDATES[name];
    if (!pool) return { candidates: [], replaceFrom: value.length };

    const endsWithSpace = /\s$/.test(value);
    const frag = endsWithSpace ? '' : parts[parts.length - 1].toLowerCase();
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
    writePrompt(cmd);
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
      { type: 'tip', text: 'type <span class="t-cmd">help</span>, then try <span class="t-cmd">find gaussian</span> or <span class="t-cmd">open reading</span>.' },
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

  // click anywhere on the terminal -> focus input
  document.getElementById('terminal').addEventListener('click', (e) => {
    if (window.getSelection().toString().length === 0) input.focus();
  });
  // auto-focus on load (but not aggressively if user is scrolling)
  setTimeout(() => { input.focus(); syncCaret(); }, 600);
})();
