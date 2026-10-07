/*
 * tailcar.js — the car you're following, plus two page-level driving touches.
 *
 *  1. Tail car (bottom right): the rear of a car with working turn signals.
 *     Left blinker = previous page (history back), right blinker = next page
 *     (history forward). The plate shows where you are.
 *  2. Lane change: pages swap with a sideways slide (cross-document view
 *     transitions; the direction is picked in an inline <head> script on
 *     `pagereveal`, see _layouts/default.html). The blinker you used keeps
 *     blinking on the page you land on.
 *  3. Night drive: switching to dark lets dusk fall from the bottom and flicks
 *     the headlights on; switching back lets the sun come up from the top.
 *
 * Reduced motion: no blinking, no slides, no beams; everything still works.
 */
(function () {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const root = document.documentElement;
  const NAV = window.navigation;            // Navigation API (Chromium); optional
  const KEY = 'tail:blink';

  // ─── 1. tail car ──────────────────────────────────────────────────────────
  function plateText() {
    if (document.querySelector('.reroute')) return '404';
    const parts = location.pathname.split('/').filter(Boolean);
    if (!parts.length) return 'HOME';
    const last = parts[parts.length - 1].replace(/\.html$/, '');
    return (last.split('-')[0] || last).toUpperCase().slice(0, 9);
  }

  const CAR = `
    <svg class="tailcar-svg" viewBox="0 0 140 92" aria-hidden="true">
      <ellipse class="tc-shadow" cx="70" cy="86" rx="60" ry="4"/>
      <rect class="tc-tyre" x="15" y="64" width="22" height="20" rx="4"/>
      <rect class="tc-tyre" x="103" y="64" width="22" height="20" rx="4"/>
      <path class="tc-paint" d="M34 38 L42 14 Q44 9 50 9 H90 Q96 9 98 14 L106 38 Z"/>
      <path class="tc-glass" d="M41 34 L47 17 Q48 14 52 14 H88 Q92 14 93 17 L99 34 Z"/>
      <rect class="tc-paint" x="7" y="34" width="126" height="40" rx="13"/>
      <rect class="tc-bar" x="13" y="43" width="114" height="9" rx="4.5"/>
      <rect class="tc-tail" x="35" y="43" width="22" height="9" rx="2"/>
      <rect class="tc-tail" x="83" y="43" width="22" height="9" rx="2"/>
      <rect class="tc-lamp tc-lamp--left" x="13" y="43" width="20" height="9" rx="4.5"/>
      <rect class="tc-lamp tc-lamp--right" x="107" y="43" width="20" height="9" rx="4.5"/>
      <path class="tc-arrow" d="M18.5 47.5 L24 44.6 V50.4 Z"/>
      <path class="tc-arrow" d="M121.5 47.5 L116 44.6 V50.4 Z"/>
      <rect class="tc-plate" x="49" y="56" width="42" height="13" rx="2.5"/>
      <text class="tc-plate-txt" x="70" y="65.6" text-anchor="middle"></text>
      <rect class="tc-bumper" x="12" y="71" width="116" height="3" rx="1.5"/>
    </svg>`;

  function tailcar() {
    const car = document.createElement('nav');
    car.className = 'tailcar';
    car.setAttribute('aria-label', 'Page history');
    car.innerHTML = CAR +
      '<button type="button" class="blinker blinker--left" data-tip="← back" aria-label="Turn left: previous page"></button>' +
      '<button type="button" class="blinker blinker--right" data-tip="forward →" aria-label="Turn right: next page"></button>' +
      '<span class="tailcar-hint" role="status" aria-live="polite"></span>';
    document.body.appendChild(car);
    car.querySelector('.tc-plate-txt').textContent = plateText();
    const hint = car.querySelector('.tailcar-hint');

    // dim a blinker when there's nowhere to go that way (only knowable with the Navigation API)
    const refresh = () => {
      if (!NAV) return;
      car.classList.toggle('no-left', !NAV.canGoBack && location.pathname === '/');
      car.classList.toggle('no-right', !NAV.canGoForward);
    };
    refresh();

    const blink = (side, times = 3) => {
      if (reduced) return;
      car.classList.remove('blink-left', 'blink-right');
      void car.offsetWidth;                                  // restart the animation
      car.style.setProperty('--blinks', times);
      car.classList.add('blink-' + side);
    };
    car.addEventListener('animationend', (e) => {
      if (e.animationName === 'tc-blink') car.classList.remove('blink-left', 'blink-right');
    });
    const say = (msg) => {
      hint.textContent = msg;
      car.classList.add('hinting');
      clearTimeout(say.t);
      say.t = setTimeout(() => car.classList.remove('hinting'), 1400);
    };
    const leave = (side, go) => {
      blink(side);
      try { sessionStorage.setItem(KEY, side); } catch (e) {}
      setTimeout(go, reduced ? 0 : 380);                     // let the blinker flash before we turn
    };

    car.querySelector('.blinker--left').addEventListener('click', () => {
      const canBack = NAV ? NAV.canGoBack : history.length > 1;
      if (canBack) leave('left', () => history.back());
      else if (location.pathname !== '/') leave('left', () => { location.href = '/'; });
      else { blink('left', 1); say('start of the route'); }
    });
    car.querySelector('.blinker--right').addEventListener('click', () => {
      if (NAV && !NAV.canGoForward) { blink('right', 1); say('no lane ahead'); return; }
      leave('right', () => history.forward());
      // without the Navigation API we can't know; if nothing happened, say so
      if (!NAV) setTimeout(() => { if (document.visibilityState === 'visible') say('no lane ahead'); }, 900);
    });

    // keep blinking on arrival: the turn you took carries over to this page
    let carried = null;
    try { carried = sessionStorage.getItem(KEY); sessionStorage.removeItem(KEY); } catch (e) {}
    if (carried === 'left' || carried === 'right') blink(carried, 2);
  }

  // ─── 3. night drive ───────────────────────────────────────────────────────
  function headlights() {
    const h = document.createElement('div');
    h.className = 'headlights';
    h.setAttribute('aria-hidden', 'true');
    h.addEventListener('animationend', () => h.remove());
    document.body.appendChild(h);
  }

  // Called by the theme toggle (and the terminal's `theme`) with the theme to
  // switch to and a function that applies it.
  window.fxThemeSwitch = function (next, apply) {
    if (reduced || !document.startViewTransition) { apply(); return; }
    const cls = next === 'dark' ? 'vt-night' : 'vt-day';
    root.classList.add(cls);
    const vt = document.startViewTransition(apply);
    vt.finished.finally(() => root.classList.remove(cls));
    if (next === 'dark') vt.ready.then(() => setTimeout(headlights, 420)).catch(() => {});
  };

  tailcar();
})();
