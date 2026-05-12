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

  // ---- TOC: build from h2/h3 in .post-content ------------------------------
  const tocEl = document.getElementById('toc-target');
  const content = document.querySelector('.post-content');
  if (!tocEl || !content) return;

  const headings = content.querySelectorAll('h2, h3');
  if (headings.length === 0) {
    const aside = tocEl.closest('.toc-aside');
    if (aside) aside.style.display = 'none';
    return;
  }

  const slugify = (s) =>
    s.toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-');

  const tocItems = [];
  headings.forEach((h) => {
    if (!h.id) h.id = slugify(h.textContent);
    const link = document.createElement('a');
    link.href = '#' + h.id;
    link.textContent = h.textContent;
    link.classList.add('lvl-' + h.tagName.charAt(1));
    tocEl.appendChild(link);
    tocItems.push({ link, target: h });
  });

  // ---- Scroll-spy: highlight current heading -------------------------------
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
})();
