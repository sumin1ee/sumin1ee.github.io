/*
 * Playful ML/CV and driving touches: diffusion section titles, a BEV online-map
 * HUD, easter eggs and an odometer (plus a flow-matching portrait, a LiDAR sweep
 * and a status ticker that only switch on where their elements exist).
 * Motion-heavy pieces bail out on reduced-motion.
 */
(function () {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ─── 3. diffusion titles ────────────────────────────────────────────────
  // Section titles start as glyph noise and denoise to text as they scroll in,
  // with a DDPM-style timestep counter ticking t=1000 → 0 beside them.
  function diffusionTitles() {
    if (reduced) return;
    const NOISE = '░▒▓#%&@$*+=?/\\<>~^01';
    const T = 28; // frames
    const titles = document.querySelectorAll('.section-title');

    const run = (el) => {
      const text = el.textContent;
      el.setAttribute('aria-label', text);
      // each char denoises at its own random step; spaces stay spaces
      const at = [...text].map((ch) => (ch === ' ' ? -1 : Math.random() * 0.85));
      const tag = document.createElement('span');
      tag.className = 'diff-t';
      el.insertAdjacentElement('afterend', tag);
      let f = 0;
      const step = () => {
        const p = f / T; // 0 = pure noise, 1 = clean
        el.textContent = [...text].map((ch, i) =>
          at[i] < 0 || p >= at[i] + 0.15 ? ch : NOISE[(Math.random() * NOISE.length) | 0]
        ).join('');
        tag.textContent = `t=${Math.round((1 - p) * 1000)}`;
        if (f++ < T) setTimeout(step, 38);
        else { el.textContent = text; tag.classList.add('done'); }
      };
      step();
    };

    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { io.unobserve(e.target); run(e.target); } });
    }, { threshold: 0.6 });
    titles.forEach((t) => io.observe(t));
  }

  // ─── 4. flow-matching portrait ──────────────────────────────────────────
  // On hover (and once on load) the photo re-generates: pixels travel along
  // straight rectified-flow paths x_t = (1-t)·x0 + t·x1 from gaussian noise.
  function flowPortrait() {
    const wrap = document.querySelector('.hero-portrait');
    const img = wrap && wrap.querySelector('img');
    if (!img || reduced) return;

    const size = () => wrap.getBoundingClientRect().width;
    const canvas = document.createElement('canvas');
    canvas.className = 'flow-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    wrap.appendChild(canvas);
    const tag = document.createElement('span');
    tag.className = 'flow-tag';
    wrap.appendChild(tag);
    const ctx = canvas.getContext('2d');

    let pts = null, S = 0, dpr = 1, playing = false;
    const gauss = () => {
      let u = 0, v = 0;
      while (!u) u = Math.random();
      while (!v) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    function sample() {
      S = size(); dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = canvas.height = S * dpr;
      canvas.style.width = canvas.style.height = S + 'px';
      const off = document.createElement('canvas');
      off.width = off.height = S;
      const o = off.getContext('2d');
      // object-fit: cover
      const r = Math.max(S / img.naturalWidth, S / img.naturalHeight);
      const w = img.naturalWidth * r, h = img.naturalHeight * r;
      o.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
      let data;
      try { data = o.getImageData(0, 0, S, S).data; } catch (e) { return false; } // tainted canvas
      const step = 3;
      pts = [];
      for (let y = 0; y < S; y += step) {
        for (let x = 0; x < S; x += step) {
          const k = (y * S + x) * 4;
          pts.push({
            x1: x, y1: y,
            x0: S / 2 + gauss() * S * 0.22, y0: S / 2 + gauss() * S * 0.22,
            c: `rgb(${data[k]},${data[k + 1]},${data[k + 2]})`,
            trail: Math.random() < 0.025,
          });
        }
      }
      return true;
    }

    function play() {
      if (playing || !img.complete) return;
      if (!pts && !sample()) return;
      playing = true;
      wrap.classList.add('is-flowing');
      // never leave the photo hidden if frames stall (background tab, headless)
      const bail = setTimeout(() => { playing = false; wrap.classList.remove('is-flowing'); }, 2500);
      const DUR = 1100, t0 = performance.now();
      const ease = (t) => 1 - Math.pow(1 - t, 3);
      const frame = (now) => {
        const t = Math.min(1, (now - t0) / DUR), e = ease(t);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, S, S);
        // a few straight trajectories, the whole point of rectified flow
        ctx.strokeStyle = css('--accent');
        ctx.globalAlpha = 0.35 * (1 - e);
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        for (const p of pts) if (p.trail) { ctx.moveTo(p.x0, p.y0); ctx.lineTo(p.x1, p.y1); }
        ctx.stroke();
        ctx.globalAlpha = 1;
        const r = 1.2 + e * 2.2;
        for (const p of pts) {
          ctx.fillStyle = p.c;
          ctx.fillRect(p.x0 + (p.x1 - p.x0) * e - r / 2, p.y0 + (p.y1 - p.y0) * e - r / 2, r, r);
        }
        tag.textContent = `flow · t=${e.toFixed(2)}`;
        if (t < 1) requestAnimationFrame(frame);
        else {
          clearTimeout(bail);
          playing = false;
          wrap.classList.remove('is-flowing');
          // resample the noise so every replay is a new draw from N(0, I)
          pts.forEach((p) => { p.x0 = S / 2 + gauss() * S * 0.22; p.y0 = S / 2 + gauss() * S * 0.22; });
        }
      };
      requestAnimationFrame(frame);
    }

    wrap.addEventListener('pointerenter', play);
    window.addEventListener('resize', () => { pts = null; });
    const first = () => setTimeout(play, 500);
    if (img.complete) first(); else img.addEventListener('load', first, { once: true });
  }

  // ─── 5. online HD map (BEV minimap) ─────────────────────────────────────
  // A fixed bird's-eye-view HUD. Scrolling drives the ego car forward; lane
  // dividers / boundaries are "predicted" as vectorized polylines inside the
  // perception range and accumulate behind the car. Sections are crosswalks.
  function onlineMap() {
    if (!window.matchMedia('(min-width: 1100px)').matches) return;
    if (document.querySelector('.post-content')) return;   // not on posts: it would sit on the table of contents

    const hud = document.createElement('aside');
    hud.className = 'bev-hud';
    hud.setAttribute('aria-hidden', 'true');
    hud.innerHTML =
      '<div class="bev-head"><span>online map · BEV</span><button type="button" class="bev-min" tabindex="-1">–</button></div>' +
      '<canvas class="bev-canvas"></canvas>' +
      '<div class="bev-foot"><span class="bev-sec"></span><span class="bev-n"></span></div>';
    document.body.appendChild(hud);
    const canvas = hud.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const secEl = hud.querySelector('.bev-sec');
    const nEl = hud.querySelector('.bev-n');
    hud.querySelector('.bev-min').addEventListener('click', () => hud.classList.toggle('is-min'));

    const W = 176, H = 212, dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';

    const PX = 2.0;          // bev px per "meter"
    const K = 0.09;          // meters travelled per scrolled px
    const EGO_Y = H * 0.72;  // ego position on the canvas
    const RANGE = 62;        // perception range ahead (m)
    const LANE = 8;          // lane width (m, exaggerated for legibility)

    // road centerline as a smooth function of distance s
    const cx = (s) => 14 * Math.sin(s / 60) + 6 * Math.sin(s / 23 + 1.3);
    const toScreen = (s, off, egoS) => [W / 2 + (cx(s) - cx(egoS) + off) * PX, EGO_Y - (s - egoS) * PX];

    // sections → crosswalks at their document position
    let crossings = [];
    const layout = () => {
      const els = [...document.querySelectorAll('.section, .hero')].filter((el) => el.offsetParent !== null);
      crossings = els.map((el) => {
        const t = el.querySelector('.section-title');
        return {
          s: (el.getBoundingClientRect().top + window.scrollY) * K,
          name: t ? (t.getAttribute('aria-label') || t.textContent).trim()
                  : el.querySelector('.terminal') ? '~/terminal' : 'about',
        };
      }).sort((a, b) => a.s - b.s);
    };
    // a few other agents parked along the road (meters, lane offset)
    const agents = [[60, LANE], [140, -LANE], [230, LANE], [320, LANE * 2], [410, -LANE], [520, LANE]];
    let seen = 0; // furthest s the map has been built to

    function draw() {
      const egoS = (window.scrollY + window.innerHeight * 0.35) * K; // ego sits 35% down the viewport
      seen = Math.max(seen, egoS + RANGE);
      const accent = css('--accent'), ink = css('--ink-2'), warn = css('--accent-warn');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // grid
      ctx.strokeStyle = css('--border'); ctx.lineWidth = 1;
      const g = 20, shift = (egoS * PX) % g;
      ctx.beginPath();
      for (let y = shift; y < H; y += g) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
      for (let x = (W / 2) % g; x < W; x += g) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
      ctx.stroke();

      const s0 = egoS - (H - EGO_Y) / PX - 5, s1 = Math.min(seen, egoS + EGO_Y / PX);
      let instances = 0;
      // polyline with vertex dots; points near the range edge jitter like fresh predictions
      const poly = (off, color, dash, width) => {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        const verts = [];
        for (let s = s0; s <= s1; s += 3) {
          const fresh = Math.max(0, (s - (egoS + RANGE * 0.6)) / (RANGE * 0.4));
          const [x, y] = toScreen(s, off + (Math.random() - 0.5) * fresh * 2.4, egoS);
          verts.push([x, y, fresh]);
          s === s0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        verts.forEach(([x, y, f], i) => { if (i % 2 === 0) { ctx.globalAlpha = 0.9 - f * 0.5; ctx.fillRect(x - 1.2, y - 1.2, 2.4, 2.4); } });
        ctx.globalAlpha = 1;
        instances++;
      };
      poly(-LANE * 1.5, ink, [], 1.4);          // road boundaries
      poly(LANE * 2.5, ink, [], 1.4);
      poly(-LANE * 0.5, accent, [5, 4], 1.2);   // lane dividers
      poly(LANE * 0.5, accent, [5, 4], 1.2);
      poly(LANE * 1.5, accent, [5, 4], 1.2);

      // crosswalks + section labels
      let current = crossings[0];
      ctx.font = '9px "JetBrains Mono", monospace';
      crossings.forEach((c) => {
        if (c.s <= egoS) current = c;
        if (c.s < s0 || c.s > s1) return;
        instances++;
        const [xl, y] = toScreen(c.s, -LANE * 1.5, egoS);
        const [xr] = toScreen(c.s, LANE * 2.5, egoS);
        ctx.fillStyle = warn; ctx.globalAlpha = 0.55;
        for (let x = xl + 2; x < xr - 2; x += 5) ctx.fillRect(x, y - 4, 2.5, 8);
        ctx.globalAlpha = 1;
        const tag = c.name.toLowerCase();
        ctx.fillText(tag.length > 12 ? tag.slice(0, 11) + '…' : tag, Math.min(xr + 3, W - 60), y + 3);
      });

      // other agents: little boxes with a heading tick
      agents.forEach(([s, off]) => {
        if (s < s0 || s > s1) return;
        instances++;
        const [x, y] = toScreen(s, off, egoS);
        ctx.strokeStyle = ink; ctx.lineWidth = 1.2;
        ctx.strokeRect(x - 3.5, y - 7, 7, 14);
        ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x, y - 11); ctx.stroke();
      });

      // perception range fan
      ctx.fillStyle = accent; ctx.globalAlpha = 0.07;
      ctx.beginPath(); ctx.moveTo(W / 2, EGO_Y);
      ctx.arc(W / 2, EGO_Y, RANGE * PX, -Math.PI / 2 - 0.55, -Math.PI / 2 + 0.55);
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;

      // ego
      ctx.fillStyle = accent;
      ctx.fillRect(W / 2 - 4, EGO_Y - 8, 8, 16);
      ctx.fillStyle = css('--bg-0');
      ctx.fillRect(W / 2 - 2.5, EGO_Y - 5, 5, 3);

      secEl.textContent = current ? '→ ' + current.name.toLowerCase() : '';
      nEl.textContent = window.__odo ? `${instances} inst · ${window.__odo.replace(/^odometer /, '')}` : `${instances} inst`;
    }

    let queued = false;
    const tick = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } };
    window.addEventListener('scroll', tick, { passive: true });
    window.addEventListener('resize', () => { layout(); tick(); });
    // a theme switch re-tints the map
    new MutationObserver(() => { layout(); tick(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    // give titles a moment to settle (diffusion swaps their text) before labeling crossings
    layout(); draw();
    setTimeout(() => { layout(); draw(); }, 1500);
  }

  // ─── 6. status ticker ───────────────────────────────────────────────────
  // The hero "status" line types through a rotation of honest grad-student states.
  function statusTicker() {
    const row = [...document.querySelectorAll('.hero-meta-row')].find((r) => r.querySelector('.status-dot'));
    if (!row) return;
    const node = [...row.childNodes].reverse().find((n) => n.nodeType === 3 && n.textContent.trim());
    if (!node) return;
    const span = document.createElement('span');
    span.className = 'status-text';
    span.textContent = node.textContent.trim();
    node.replaceWith(' ', span);
    if (reduced) return;

    const LINES = [
      span.textContent,
      'training gaussians at 3am 🌙',
      'CUDA out of memory (again) 🫠',
      'reading arXiv instead of sleeping 📄',
      'waiting for the loss to go down 📉',
      'hunting for a lab coffee refill ☕',
    ];
    let i = 0;
    const type = (text, k = 0) => {
      span.textContent = text.slice(0, k);
      if (k < text.length) setTimeout(() => type(text, k + 1), 32);
      else setTimeout(erase, 3200);
    };
    const erase = () => {
      const t = span.textContent;
      if (t.length) { span.textContent = [...t].slice(0, -1).join(''); setTimeout(erase, 14); }
      else { i = (i + 1) % LINES.length; type(LINES[i]); }
    };
    setTimeout(erase, 4000);
  }

  // ─── 7. splat party (easter egg) ────────────────────────────────────────
  // Konami code, typing "splat", or clicking an "accepted" tag throws a burst
  // of gaussian confetti. Exposed as window.fxParty for the blog terminal.
  function splatParty() {
    const party = (msg) => {
      if (reduced) return;
      const c = document.createElement('canvas');
      c.className = 'party-canvas';
      c.setAttribute('aria-hidden', 'true');
      document.body.appendChild(c);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = innerWidth, H = innerHeight;
      c.width = W * dpr; c.height = H * dpr;
      const g = c.getContext('2d');
      g.scale(dpr, dpr);
      const colors = [css('--accent'), css('--accent'), css('--ink-0'), css('--ink-2')];
      const ps = Array.from({ length: 160 }, () => ({
        x: W / 2 + (Math.random() - 0.5) * 80, y: H * 0.55,
        vx: (Math.random() - 0.5) * 16, vy: -8 - Math.random() * 12,
        r: 3 + Math.random() * 7, rot: Math.random() * Math.PI, ecc: 0.35 + Math.random() * 0.65,
        c: colors[(Math.random() * colors.length) | 0],
      }));
      let f = 0;
      const frame = () => {
        g.clearRect(0, 0, W, H);
        for (const p of ps) {
          p.vy += 0.35; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.rot += 0.05;
          // each confetto is an anisotropic gaussian: soft ellipse, random orientation
          g.save(); g.translate(p.x, p.y); g.rotate(p.rot); g.scale(1, p.ecc);
          const grad = g.createRadialGradient(0, 0, 0, 0, 0, p.r);
          grad.addColorStop(0, p.c); grad.addColorStop(1, 'transparent');
          g.fillStyle = grad; g.globalAlpha = Math.max(0, 1 - f / 140);
          g.beginPath(); g.arc(0, 0, p.r, 0, Math.PI * 2); g.fill();
          g.restore();
        }
        if (++f < 140) requestAnimationFrame(frame); else c.remove();
      };
      frame();
      if (msg) toast(msg);
    };

    const toast = (msg) => {
      const t = document.createElement('div');
      t.className = 'fx-toast';
      t.setAttribute('role', 'status');
      t.textContent = msg;
      document.body.appendChild(t);
      requestAnimationFrame(() => t.classList.add('on'));
      setTimeout(() => { t.classList.remove('on'); setTimeout(() => t.remove(), 400); }, 2600);
    };

    window.fxParty = party;

    const KONAMI = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
    let k = 0, typed = '';
    document.addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea, [contenteditable]')) return;
      k = e.key === KONAMI[k] ? k + 1 : e.key === KONAMI[0] ? 1 : 0;
      if (k === KONAMI.length) { k = 0; party('🎮 achievement unlocked: konami splat'); }
      typed = (typed + e.key).slice(-5).toLowerCase();
      if (typed === 'splat') party('🫧 splat party!');
    });
    document.addEventListener('click', (e) => {
      if (e.target.closest('.news-tag--accepted, .status-pill--accept')) party('🎉 accepted!! (still not over it)');
    });
  }

  // ─── 8. little extras ───────────────────────────────────────────────────
  function extras() {
    // tab title pleads when you switch away
    const title = document.title;
    document.addEventListener('visibilitychange', () => {
      document.title = document.hidden ? 'come back, the gaussians miss you 🥺' : title;
    });
    // a hello for whoever opens devtools
    console.log(
      '%c hey there, fellow dev 👀 %c\nif you are reading the source, we should talk: ' +
        ((document.querySelector('a[href^="mailto:"]') || {}).href || 'github.com/sumin1ee').replace('mailto:', ''),
      `background:${css('--accent')};color:${css('--accent-ink')};font:600 13px monospace;padding:4px 6px;border-radius:4px`,
      'font:12px monospace'
    );
  }

  // ─── 9. LiDAR sweep (cv hero) ──────────────────────────────────────────
  // A spinning LiDAR sits where the portrait is and paints a synthetic street
  // in BEV: ground rings, high-intensity lane markings, parked cars (which get
  // faint boxes when the beam hits them).
  function lidarHero() {
    const hero = document.querySelector('.hero');
    if (!hero || !hero.querySelector('.hero-name') || reduced) return;
    hero.classList.add('has-lidar');
    const canvas = document.createElement('canvas');
    canvas.className = 'lidar-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    hero.prepend(canvas);
    const ctx = canvas.getContext('2d');

    const SQ = 0.5;                       // squash y: a tilted-ground look
    let W, H, dpr, ox, oy, R = 300, pts = [], cars = [];

    function build() {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = r.width; H = r.height;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const hr = hero.getBoundingClientRect();
      const p = hero.querySelector('.hero-portrait');
      const pr = p && p.getBoundingClientRect();
      // sensor origin = portrait center, else upper right
      ox = pr && pr.width ? pr.left + pr.width / 2 - r.left : W * 0.8;
      oy = pr && pr.width ? pr.top + pr.height / 2 - r.top : H * 0.3;

      R = Math.min(300, Math.max(W, H) * 0.45);
      const LANE = 30;
      pts = []; cars = [];
      const inCar = (x, y) => cars.some((c) => Math.abs(x - c.x) < c.w / 2 + 2 && Math.abs(y - c.y) < c.h / 2 + 2);
      for (let i = 0; i < 4; i++) {
        const lane = [-1.5, -0.5, 0.5, 1.5][(Math.random() * 4) | 0];
        const y = (Math.random() < 0.5 ? -1 : 1) * (60 + Math.random() * R * 0.6);
        cars.push({ x: lane * LANE, y, w: 20, h: 42, score: (0.86 + Math.random() * 0.13).toFixed(2),
                    cls: Math.random() < 0.8 ? 'car' : Math.random() < 0.5 ? 'truck' : 'ped' });
      }
      const add = (x, y, kind) => {
        const d = Math.hypot(x, y);
        if (d < 18 || d > R) return;
        pts.push({ x, y, a: Math.atan2(y, x), d, kind });
      };
      // ground returns: concentric rings, sparser with range
      for (let k = 1; k * 22 < R; k++) {
        const rr = 14 + k * 22, n = Math.floor((2 * Math.PI * rr) / (5 + k * 0.4));
        for (let i = 0; i < n; i++) {
          const a = (i / n) * Math.PI * 2 + k * 0.1;
          const x = rr * Math.cos(a) + (Math.random() - 0.5) * 2, y = rr * Math.sin(a) + (Math.random() - 0.5) * 2;
          if (!inCar(x, y) && Math.abs(x) < LANE * 2 + 60) add(x, y, 0);
          else if (Math.abs(x) >= LANE * 2 + 60 && Math.random() < 0.35) add(x, y, 0);
        }
      }
      // lane markings: high intensity. dashed dividers, solid boundaries
      for (const m of [-2, -1, 0, 1, 2]) {
        for (let y = -R; y < R; y += 4) {
          if (Math.abs(m) < 2 && ((y % 34) + 34) % 34 > 20) continue;
          if (!inCar(m * LANE, y)) add(m * LANE + (Math.random() - 0.5), y, 1);
        }
      }
      // cars: points on their outline
      cars.forEach((c, ci) => {
        for (let t = 0; t < 1; t += 0.035) {
          add(c.x - c.w / 2 + t * c.w, c.y + c.h / 2, 2); add(c.x - c.w / 2 + t * c.w, c.y - c.h / 2, 2);
          add(c.x - c.w / 2, c.y - c.h / 2 + t * c.h, 2); add(c.x + c.w / 2, c.y - c.h / 2 + t * c.h, 2);
        }
        c.a = Math.atan2(c.y, c.x);
      });
    }

    let theta = 0, raf = null, visible = true, last = performance.now();
    function frame(now) {
      const dt = Math.min(64, now - last); last = now;
      theta = (theta + dt * 0.0026) % (Math.PI * 2);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      // ground = gray returns, lane paint = bright, detected objects = accent
      const cols = [css('--ink-2'), css('--ink-0'), css('--accent'), css('--ink-2')];
      const base = [0.7, 1, 0.9, 0];
      for (const p of pts) {
        const since = (theta - p.a + Math.PI * 4) % (Math.PI * 2);      // radians since the beam passed
        const I = base[p.kind] * Math.exp(-since * 0.5) * (1 - p.d / R);
        if (I < 0.02) continue;
        ctx.globalAlpha = I; ctx.fillStyle = cols[p.kind];
        ctx.fillRect(ox + p.x - 1, oy + p.y * SQ - 1, 2, 2);
      }
      // boxes on freshly scanned objects
      ctx.font = '10px "JetBrains Mono", monospace';
      for (const c of cars) {
        const since = (theta - c.a + Math.PI * 4) % (Math.PI * 2);
        const a = Math.exp(-since * 0.7);
        if (a < 0.05) continue;
        ctx.globalAlpha = a * 0.45; ctx.strokeStyle = cols[2]; ctx.lineWidth = 1;
        const x = ox + c.x - c.w / 2, y = oy + (c.y - c.h / 2) * SQ;
        ctx.strokeRect(x, y, c.w, c.h * SQ);
      }
      // the beam
      ctx.globalAlpha = 0.1; ctx.strokeStyle = cols[2]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ox, oy);
      ctx.lineTo(ox + Math.cos(theta) * R, oy + Math.sin(theta) * R * SQ); ctx.stroke();
      ctx.globalAlpha = 1;
      raf = visible && !document.hidden ? requestAnimationFrame(frame) : null;
    }
    const wake = () => { if (!raf && visible && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); } };
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; wake(); }).observe(hero);
    document.addEventListener('visibilitychange', wake);
    let rt;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(build, 150); });
    (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => { build(); wake(); });
  }

  // ─── 10. odometer ───────────────────────────────────────────────────────
  // Scrolling is driving. Distance accumulates across visits (per browser).
  function odometer() {
    const KEY = 'odo_m', M_PER_PX = 0.05;
    let m = 0;
    try { m = parseFloat(localStorage.getItem(KEY)) || 0; } catch (e) {}
    let y = window.scrollY, saveT;
    const els = () => document.querySelectorAll('[data-odo]');
    const show = () => {
      const txt = m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(2)} km`;
      els().forEach((el) => { el.textContent = `odometer ${txt}`; });
      window.__odo = txt;
    };
    // the little ego car that rides the road rail (styled/hidden in CSS)
    const car = document.createElement('div');
    car.className = 'rail-car';
    car.setAttribute('aria-hidden', 'true');
    document.body.appendChild(car);

    window.addEventListener('scroll', () => {
      m += Math.abs(window.scrollY - y) * M_PER_PX; y = window.scrollY;
      show();
      clearTimeout(saveT);
      saveT = setTimeout(() => { try { localStorage.setItem(KEY, String(m)); } catch (e) {} }, 400);
    }, { passive: true });
    show();
  }

  lidarHero();
  diffusionTitles();
  flowPortrait();
  onlineMap();
  statusTicker();
  splatParty();
  odometer();
  extras();
})();
