(function () {
  // ---- Reading progress bar ------------------------------------------------
  const bar = document.getElementById('reading-progress-bar');
  if (bar) {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      const pct = max > 0 ? Math.min(100, Math.max(0, (h.scrollTop / max) * 100)) : 0;
      bar.style.width = pct + '%';
    };
    document.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  const content = document.querySelector('.post-content');

  // ---- Language toggle (KOR / ENG) ----------------------------------------
  // A bilingual post wraps each language in <div class="lang-block" lang="ko|en">.
  // We show one block at a time and remember the choice in localStorage.
  const langButtons = Array.from(document.querySelectorAll('.lang-btn'));
  const langBlocks = content
    ? Array.from(content.querySelectorAll('.lang-block'))
    : [];

  let currentLang = null;

  const applyLang = (lang) => {
    currentLang = lang;
    langBlocks.forEach((b) => {
      b.hidden = b.getAttribute('lang') !== lang;
    });
    langButtons.forEach((btn) => {
      const on = btn.getAttribute('data-lang') === lang;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    document.documentElement.setAttribute('data-post-lang', lang);
    buildToc();
  };

  if (langButtons.length && langBlocks.length) {
    const langs = langBlocks.map((b) => b.getAttribute('lang'));
    let stored = null;
    try {
      stored = localStorage.getItem('post-lang');
    } catch (e) {}
    const initial = langs.includes(stored) ? stored : langs[0];

    langButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const lang = btn.getAttribute('data-lang');
        applyLang(lang);
        try {
          localStorage.setItem('post-lang', lang);
        } catch (e) {}
      });
    });

    applyLang(initial);
  }

  // ---- TOC: build from h2/h3 in the visible content -----------------------
  const tocEl = document.getElementById('toc-target');

  function buildToc() {
    if (!tocEl || !content) return;

    // Scope headings to the active language block when bilingual.
    const scope =
      langBlocks.length && currentLang
        ? content.querySelector('.lang-block[lang="' + currentLang + '"]')
        : content;
    const aside = tocEl.closest('.toc-aside');

    tocEl.innerHTML = '';

    const headings = scope ? scope.querySelectorAll('h2, h3') : [];
    if (!headings || headings.length === 0) {
      if (aside) aside.style.display = 'none';
      return;
    }
    if (aside) aside.style.display = '';

    const slugify = (s) =>
      s
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .trim()
        .replace(/\s+/g, '-');

    const tocItems = [];
    const seen = {};
    headings.forEach((h) => {
      if (!h.id) {
        let base = slugify(h.textContent) || 'section';
        if (seen[base] != null) {
          seen[base] += 1;
          base = base + '-' + seen[base];
        } else {
          seen[base] = 0;
        }
        h.id = base;
      }
      const link = document.createElement('a');
      link.href = '#' + h.id;
      link.textContent = h.textContent;
      link.classList.add('lvl-' + h.tagName.charAt(1));
      tocEl.appendChild(link);
      tocItems.push({ link, target: h });
    });

    // ---- Scroll-spy: highlight current heading -----------------------------
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const id = e.target.id;
          const item = tocItems.find((t) => t.target.id === id);
          if (!item) return;
          if (e.isIntersecting) {
            tocItems.forEach((t) => t.link.classList.remove('is-active'));
            item.link.classList.add('is-active');
          }
        });
      },
      { rootMargin: '-100px 0px -65% 0px', threshold: 0 }
    );
    headings.forEach((h) => observer.observe(h));
  }

  // For non-bilingual posts, build the TOC once on load.
  if (!langBlocks.length) buildToc();

  // MathJax can rewrite heading text while typesetting; rebuild the TOC once
  // it finishes so headings with inline math still slug + link correctly.
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(buildToc).catch(function () {});
  }
})();
