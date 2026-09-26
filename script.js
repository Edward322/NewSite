/* ==========================================================================
   NOVA — слайды, переходы и эффекты. Без зависимостей.
   ========================================================================== */

(() => {
  'use strict';

  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  const easeOutExpo = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;

  const pointer = { x: innerWidth / 2, y: innerHeight / 2 };
  addEventListener('pointermove', (e) => {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
  }, { passive: true });

  /* ---------- Общий цикл анимации ---------- */

  const ticker = new Set();
  const loop = (time) => {
    ticker.forEach((fn) => fn(time));
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  /* ==========================================================================
     Разбивка текста на слова / буквы
     ========================================================================== */

  function splitText(el) {
    const mode = el.dataset.split;
    const label = el.textContent.replace(/\s+/g, ' ').trim();
    let index = 0;

    const makeUnit = (text) => {
      const c = document.createElement('span');
      c.className = 'c';
      c.textContent = text;
      c.style.setProperty('--i', index++);
      return c;
    };

    const walk = (node) => {
      [...node.childNodes].forEach((child) => {
        if (child.nodeType === Node.ELEMENT_NODE) {
          walk(child);
          return;
        }
        if (child.nodeType !== Node.TEXT_NODE) return;

        const whole = child.parentElement.closest('[data-split-whole]');
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((part) => {
          if (!part) return;
          if (/^\s+$/.test(part)) {
            frag.append(' ');
            return;
          }
          const word = document.createElement('span');
          word.className = 'w';
          if (mode === 'chars' && !whole) {
            for (const ch of part) word.append(makeUnit(ch));
          } else {
            word.append(makeUnit(part));
          }
          frag.append(word);
        });
        child.replaceWith(frag);
      });
    };

    walk(el);

    // Скринридеры читают цельный текст, а не отдельные буквы
    [...el.children].forEach((child) => child.setAttribute('aria-hidden', 'true'));
    const sr = document.createElement('span');
    sr.className = 'sr-only';
    sr.textContent = label;
    el.append(sr);
  }

  $$('[data-split]').forEach(splitText);

  /* ==========================================================================
     Слайдер
     ========================================================================== */

  const slides = $$('.slide');
  const total = slides.length;
  const DURATION = reduceMotion ? 450 : 1250;
  const EASE = 'cubic-bezier(0.76, 0, 0.24, 1)';

  let current = -1;
  let ready = false; // true после прелоадера
  let animating = false;
  const hooks = new Map(); // id → { enter, leave }

  const onSlide = (id, enter, leave) => hooks.set(id, { enter, leave });
  const inner = (slide) => $('.slide__inner', slide);

  /* Формы «шторки» для разных переходов */
  function clipFrames(type, dir) {
    const w = innerWidth;
    const h = innerHeight;

    if (type === 'circle') {
      const x = clamp(pointer.x, 0, w);
      const y = clamp(pointer.y, 0, h);
      const r = Math.hypot(Math.max(x, w - x), Math.max(y, h - y));
      return [
        { clipPath: `circle(0px at ${x}px ${y}px)` },
        { clipPath: `circle(${r}px at ${x}px ${y}px)` },
      ];
    }

    if (type === 'diagonal') {
      return dir > 0
        ? [
          { clipPath: 'polygon(0% 100%, 100% 135%, 100% 100%, 0% 100%)' },
          { clipPath: 'polygon(0% -35%, 100% 0%, 100% 100%, 0% 100%)' },
        ]
        : [
          { clipPath: 'polygon(0% 0%, 100% 0%, 100% 0%, 0% -35%)' },
          { clipPath: 'polygon(0% 0%, 100% 0%, 100% 135%, 0% 100%)' },
        ];
    }

    // «Жидкая» волна: край шторки изгибается и выпрямляется
    const steps = 14;
    const points = 24;
    const frames = [];
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const base = dir > 0 ? (1 - t) * 100 : t * 100;
      const bulge = Math.sin(Math.PI * t) * 16;
      const edge = [];
      for (let p = 0; p <= points; p++) {
        const k = Math.sin((Math.PI * p) / points);
        const y = dir > 0 ? base - bulge * k : base + bulge * k;
        edge.push(`${((p / points) * 100).toFixed(2)}% ${y.toFixed(2)}%`);
      }
      const poly = dir > 0
        ? [...edge, '100% 100%', '0% 100%']
        : ['0% 0%', '100% 0%', ...edge.reverse()];
      frames.push({ clipPath: `polygon(${poly.join(',')})`, offset: t });
    }
    return frames;
  }

  function goTo(next, { instant = false } = {}) {
    next = clamp(next, 0, total - 1);
    if (next === current || animating || (!ready && !instant)) return;
    ready = true;

    const prev = current;
    const dir = next > prev ? 1 : -1;
    const inEl = slides[next];
    const outEl = slides[prev];
    current = next;

    if (outEl) {
      hooks.get(outEl.id)?.leave?.(outEl);
      outEl.classList.remove('is-active');
      outEl.classList.add('is-leaving');
    }
    inEl.classList.add('is-active');
    inner(inEl).scrollTop = 0;
    hooks.get(inEl.id)?.enter?.(inEl);
    updateUI();

    const done = () => {
      outEl?.classList.remove('is-leaving');
      animating = false;
    };

    // Шапка и счётчик перекрашиваются, когда новый слайд закрывает экран наполовину
    const setUiTheme = () => {
      document.documentElement.dataset.ui = inEl.dataset.theme || 'dark';
    };

    if (instant || !outEl) {
      setUiTheme();
      done();
      return;
    }

    setTimeout(setUiTheme, DURATION * 0.45);

    animating = true;
    const opts = { duration: DURATION, easing: EASE };
    const type = reduceMotion ? 'fade' : inEl.dataset.transition || 'liquid';
    const anims = [];

    if (type === 'fade') {
      anims.push(inEl.animate([{ opacity: 0 }, { opacity: 1 }], opts));
    } else {
      anims.push(
        inEl.animate(clipFrames(type, dir), opts),
        inner(inEl).animate(
          [{ transform: `translateY(${dir * 16}vh)` }, { transform: 'none' }],
          opts,
        ),
        $('.slide__bg', inEl).animate(
          [{ transform: 'scale(1.25)' }, { transform: 'none' }],
          opts,
        ),
        outEl.animate(
          [
            { transform: 'none', opacity: 1 },
            { transform: `translateY(${-dir * 12}vh) scale(0.92)`, opacity: 0.25 },
          ],
          opts,
        ),
      );
    }

    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      anims.forEach((a) => a.cancel());
      done();
    };
    Promise.all(anims.map((a) => a.finished)).then(finish, finish);
    setTimeout(finish, DURATION + 400); // страховка, если вкладка была в фоне
  }

  const nextSlide = () => goTo(current + 1);
  const prevSlide = () => goTo(current - 1);

  /* ---------- Интерфейс: точки, счётчик, прогресс ---------- */

  const dotsNav = $('.dots');
  const dots = slides.map((slide, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'dots__item';
    btn.setAttribute('aria-label', `${i + 1}. ${slide.dataset.title}`);
    const label = document.createElement('span');
    label.className = 'dots__label';
    label.textContent = slide.dataset.title;
    btn.append(label);
    btn.addEventListener('click', () => goTo(i));
    dotsNav.append(btn);
    return btn;
  });

  const pad = (n) => String(n).padStart(2, '0');
  const roll = $('.hud__roll');
  slides.forEach((_, i) => {
    const span = document.createElement('span');
    span.textContent = pad(i + 1);
    roll.append(span);
  });
  $('.hud__total').textContent = pad(total);

  const hudFill = $('.hud__fill');
  const hudTitle = $('.hud__title');
  const [btnPrev, btnNext] = $$('.hud__btn');
  const navLinks = $$('.nav a');

  function updateUI() {
    const slide = slides[current];
    dots.forEach((d, i) => {
      d.classList.toggle('is-active', i === current);
      if (i === current) d.setAttribute('aria-current', 'true');
      else d.removeAttribute('aria-current');
    });
    navLinks.forEach((a) => a.classList.toggle('is-current', a.hash === `#${slide.id}`));
    roll.style.transform = `translateY(${-current * 1.2}em)`;
    hudFill.style.transform = `scaleX(${(current + 1) / total})`;
    hudTitle.textContent = slide.dataset.title;
    btnPrev.disabled = current === 0;
    btnNext.disabled = current === total - 1;
    try {
      history.replaceState(null, '', current === 0 ? location.pathname + location.search : `#${slide.id}`);
    } catch {
      // file:// в некоторых браузерах запрещает менять адрес — не страшно
    }
  }

  /* ---------- Управление ---------- */

  // Может ли содержимое слайда само прокрутиться в нужную сторону (маленькие экраны)
  function canScroll(dir) {
    const slide = slides[current];
    if (!slide) return false;
    const el = inner(slide);
    if (el.scrollHeight - el.clientHeight < 2) return false;
    return dir > 0
      ? el.scrollTop + el.clientHeight < el.scrollHeight - 2
      : el.scrollTop > 2;
  }

  // Колесо / тачпад: одно движение — один слайд, инерция игнорируется
  let wheelSum = 0;
  let wheelLast = 0;
  let wheelLocked = false;
  addEventListener('wheel', (e) => {
    if (e.ctrlKey) return; // масштабирование
    const now = performance.now();
    const gap = now - wheelLast;
    wheelLast = now;
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;

    const dir = Math.sign(e.deltaY);
    if (!dir) return;
    if (canScroll(dir)) return;
    e.preventDefault();

    // Во время перехода колесо игнорируется, а новый жест начинается только после паузы
    if (animating) return;
    if (gap > 220) {
      wheelLocked = false;
      wheelSum = 0;
    }
    if (wheelLocked) return;

    wheelSum += e.deltaY;
    if (Math.abs(wheelSum) > 35) {
      wheelLocked = true;
      wheelSum = 0;
      dir > 0 ? nextSlide() : prevSlide();
    }
  }, { passive: false });

  // Свайпы
  let touch = null;
  addEventListener('touchstart', (e) => {
    const t = e.touches[0];
    touch = { x: t.clientX, y: t.clientY, down: canScroll(1), up: canScroll(-1) };
  }, { passive: true });

  addEventListener('touchend', (e) => {
    if (!touch) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touch.x;
    const dy = t.clientY - touch.y;
    if (Math.abs(dy) > 50 && Math.abs(dy) > Math.abs(dx) * 1.2) {
      if (dy < 0 && !touch.down) nextSlide();
      if (dy > 0 && !touch.up) prevSlide();
    }
    touch = null;
  }, { passive: true });

  // Клавиатура
  addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.target.closest('input, textarea, select, [contenteditable]')) return;
    const onControl = e.target.closest('button, a');

    switch (e.key) {
      case 'ArrowDown':
      case 'PageDown':
        e.preventDefault();
        nextSlide();
        break;
      case 'ArrowUp':
      case 'PageUp':
        e.preventDefault();
        prevSlide();
        break;
      case ' ':
        if (onControl) return;
        e.preventDefault();
        e.shiftKey ? prevSlide() : nextSlide();
        break;
      case 'Home':
        e.preventDefault();
        goTo(0);
        break;
      case 'End':
        e.preventDefault();
        goTo(total - 1);
        break;
      default:
        if (/^[1-9]$/.test(e.key) && +e.key <= total) goTo(+e.key - 1);
    }
  });

  // Кнопки и якорные ссылки
  document.addEventListener('click', (e) => {
    const nav = e.target.closest('[data-nav]');
    if (nav) {
      nav.dataset.nav === 'next' ? nextSlide() : prevSlide();
      return;
    }
    const link = e.target.closest('a[href^="#"]');
    if (!link) return;
    e.preventDefault();
    const index = slides.findIndex((s) => `#${s.id}` === link.hash);
    if (index !== -1) goTo(index);
  });

  addEventListener('hashchange', () => {
    const index = slides.findIndex((s) => `#${s.id}` === location.hash);
    if (index !== -1) goTo(index);
  });

  /* ==========================================================================
     Прелоадер
     ========================================================================== */

  function runPreloader() {
    const el = $('.preloader');
    if (!el) return Promise.resolve();

    const count = $('.preloader__count span', el);
    const bar = $('.preloader__bar', el);
    const minTime = reduceMotion ? 1 : 1500;
    const start = performance.now();
    let loaded = false;

    const pageLoad = new Promise((resolve) => {
      if (document.readyState === 'complete') resolve();
      else addEventListener('load', resolve, { once: true });
    });
    Promise.race([
      Promise.all([pageLoad, document.fonts ? document.fonts.ready : null]),
      new Promise((resolve) => setTimeout(resolve, 4000)),
    ]).then(() => { loaded = true; });

    return new Promise((resolve) => {
      const tick = (now) => {
        const t = clamp((now - start) / minTime, 0, 1);
        const progress = loaded ? easeOutCubic(t) : Math.min(easeOutCubic(t), 0.9);
        count.textContent = Math.round(progress * 100);
        bar.style.setProperty('--progress', progress);

        if (loaded && t >= 1) {
          el.classList.add('is-done');
          setTimeout(() => el.remove(), 1200);
          resolve();
          return;
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  /* ==========================================================================
     Курсор
     ========================================================================== */

  function initCursor() {
    const el = $('.cursor');
    document.documentElement.classList.add('has-cursor');
    const dot = $('.cursor__dot', el);
    const ring = $('.cursor__ring', el);
    const text = $('.cursor__text', el);
    let rx = pointer.x;
    let ry = pointer.y;

    addEventListener('pointermove', () => el.classList.add('is-visible'), { passive: true });
    document.documentElement.addEventListener('pointerleave', () => el.classList.remove('is-visible'));
    addEventListener('pointerdown', () => el.classList.add('is-down'));
    addEventListener('pointerup', () => el.classList.remove('is-down'));

    document.addEventListener('pointerover', (e) => {
      const labelled = e.target.closest('[data-cursor-text]');
      const hover = e.target.closest('a, button');
      el.classList.toggle('has-text', !!labelled && !hover);
      el.classList.toggle('is-hover', !!hover);
      if (labelled) text.textContent = labelled.dataset.cursorText;
    });

    ticker.add(() => {
      rx = lerp(rx, pointer.x, 0.2);
      ry = lerp(ry, pointer.y, 0.2);
      dot.style.transform = `translate3d(${pointer.x}px, ${pointer.y}px, 0)`;
      ring.style.transform = `translate3d(${rx}px, ${ry}px, 0)`;
    });
  }

  /* ==========================================================================
     Параллакс за мышью
     ========================================================================== */

  function initParallax() {
    const layers = new Map(slides.map((s) => [s, $$('[data-depth]', s)]));
    let mx = 0;
    let my = 0;
    ticker.add(() => {
      const tx = (pointer.x / innerWidth - 0.5) * 2;
      const ty = (pointer.y / innerHeight - 0.5) * 2;
      mx = lerp(mx, tx, 0.06);
      my = lerp(my, ty, 0.06);
      const slide = slides[current];
      if (!slide) return;
      layers.get(slide).forEach((layer) => {
        const d = parseFloat(layer.dataset.depth) * 28;
        layer.style.transform = `translate3d(${(-mx * d).toFixed(2)}px, ${(-my * d).toFixed(2)}px, 0)`;
      });
    });
  }

  /* ==========================================================================
     Магнитные кнопки и 3D-наклон карточек
     ========================================================================== */

  function initMagnetic() {
    $$('.magnetic').forEach((el) => {
      const strength = parseFloat(el.dataset.strength || 0.3);
      let rect = null;
      el.addEventListener('pointerenter', () => {
        el.style.transform = '';
        rect = el.getBoundingClientRect();
      });
      el.addEventListener('pointermove', (e) => {
        if (!rect) return;
        const dx = e.clientX - (rect.left + rect.width / 2);
        const dy = e.clientY - (rect.top + rect.height / 2);
        el.style.transform = `translate(${dx * strength}px, ${dy * strength}px)`;
      });
      el.addEventListener('pointerleave', () => {
        rect = null;
        el.style.transform = '';
      });
    });
  }

  function initTilt() {
    $$('.card').forEach((card) => {
      card.addEventListener('pointermove', (e) => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width;
        const py = (e.clientY - r.top) / r.height;
        card.classList.add('is-tilting');
        card.style.setProperty('--rx', `${((0.5 - py) * 16).toFixed(2)}deg`);
        card.style.setProperty('--ry', `${((px - 0.5) * 16).toFixed(2)}deg`);
        card.style.setProperty('--mx', `${(px * 100).toFixed(1)}%`);
        card.style.setProperty('--my', `${(py * 100).toFixed(1)}%`);
      });
      card.addEventListener('pointerleave', () => {
        card.classList.remove('is-tilting');
        card.style.setProperty('--rx', '0deg');
        card.style.setProperty('--ry', '0deg');
      });
    });
  }

  /* ==========================================================================
     Частицы на главном слайде
     ========================================================================== */

  function initParticles(canvas) {
    const ctx = canvas.getContext('2d');
    let w = 0;
    let h = 0;
    let particles = [];
    let running = false;
    const LINK = 130;

    function resize() {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(clamp((w * h) / 15000, 28, 100));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.5,
      }));
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      const mouseActive = finePointer;

      for (const p of particles) {
        if (running && !reduceMotion) {
          p.x += p.vx;
          p.y += p.vy;
          if (p.x < -10) p.x = w + 10;
          if (p.x > w + 10) p.x = -10;
          if (p.y < -10) p.y = h + 10;
          if (p.y > h + 10) p.y = -10;

          if (mouseActive) {
            const dx = p.x - pointer.x;
            const dy = p.y - pointer.y;
            const d = Math.hypot(dx, dy);
            if (d < 150 && d > 0.1) {
              const f = (150 - d) / 150;
              p.x += (dx / d) * f * 2.2;
              p.y += (dy / d) * f * 2.2;
            }
          }
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(230, 225, 255, 0.75)';
        ctx.fill();
      }

      ctx.lineWidth = 0.7;
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < LINK) {
            ctx.strokeStyle = `rgba(160, 140, 255, ${((1 - d / LINK) * 0.35).toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
        if (mouseActive) {
          const d = Math.hypot(a.x - pointer.x, a.y - pointer.y);
          if (d < 200) {
            ctx.strokeStyle = `rgba(255, 120, 180, ${((1 - d / 200) * 0.5).toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(pointer.x, pointer.y);
            ctx.stroke();
          }
        }
      }
    }

    ticker.add(() => {
      if (running && !reduceMotion) draw();
    });

    let resizeTimer;
    addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resize, 150);
    });
    resize();

    return {
      start: () => { running = true; },
      stop: () => { running = false; },
    };
  }

  /* ==========================================================================
     Счётчики
     ========================================================================== */

  function runCounters(slide) {
    $$('[data-count]', slide).forEach((el, i) => {
      const target = parseInt(el.dataset.count, 10);
      if (reduceMotion) {
        el.textContent = target;
        return;
      }
      el.textContent = '0';
      const start = performance.now() + 450 + i * 120;
      const duration = 2200;
      const step = (now) => {
        if (!slide.classList.contains('is-active')) return;
        const t = clamp((now - start) / duration, 0, 1);
        el.textContent = Math.round(target * easeOutExpo(t));
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ==========================================================================
     3D-карусель работ
     ========================================================================== */

  function initRing(stage) {
    const ring = $('.ring', stage);
    const cards = $$('.work-card', ring);
    const n = cards.length;
    const step = 360 / n;
    let rot = 0;
    let target = 0;
    let dragging = false;
    let hovering = false;
    let lastX = 0;
    let velocity = 0;
    let lastAction = performance.now();
    let active = false;

    cards.forEach((card, i) => card.style.setProperty('--i', i));
    ring.style.setProperty('--step', `${step}deg`);

    const layout = () => {
      const r = (ring.offsetWidth / 2 / Math.tan(Math.PI / n)) * 1.18;
      ring.style.setProperty('--r', `${Math.round(r)}px`);
    };
    layout();
    addEventListener('resize', layout);

    const snap = (value) => Math.round(value / step) * step;
    const shift = (delta) => {
      target = snap(target) + delta * step;
      lastAction = performance.now();
    };

    stage.addEventListener('pointerenter', () => { hovering = true; });
    stage.addEventListener('pointerleave', () => { hovering = false; });

    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      lastX = e.clientX;
      velocity = 0;
      stage.setPointerCapture(e.pointerId);
      stage.classList.add('is-dragging');
    });

    stage.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      lastX = e.clientX;
      velocity = dx * 0.28;
      target += velocity;
    });

    const release = () => {
      if (!dragging) return;
      dragging = false;
      stage.classList.remove('is-dragging');
      target = snap(target + velocity * 10);
      lastAction = performance.now();
    };
    stage.addEventListener('pointerup', release);
    stage.addEventListener('pointercancel', release);

    $$('[data-ring]').forEach((btn) => {
      btn.addEventListener('click', () => shift(btn.dataset.ring === 'next' ? -1 : 1));
    });

    addEventListener('keydown', (e) => {
      if (!active) return;
      if (e.key === 'ArrowLeft') shift(1);
      if (e.key === 'ArrowRight') shift(-1);
    });

    ticker.add((now) => {
      if (!active) return;
      if (!dragging && !hovering && !reduceMotion && now - lastAction > 3200) shift(-1);

      rot = reduceMotion ? target : lerp(rot, target, dragging ? 0.3 : 0.075);
      ring.style.transform = `translateZ(calc(var(--r) * -1)) rotateY(${rot.toFixed(3)}deg)`;

      cards.forEach((card, i) => {
        const angle = ((((i * step + rot) % 360) + 540) % 360) - 180;
        const facing = (Math.cos((angle * Math.PI) / 180) + 1) / 2;
        card.style.opacity = (0.15 + 0.85 * facing ** 1.5).toFixed(3);
      });
    });

    return {
      start: () => {
        active = true;
        lastAction = performance.now();
      },
      stop: () => { active = false; },
    };
  }

  /* ==========================================================================
     Запуск
     ========================================================================== */

  const particles = initParticles($('.particles'));
  onSlide('home', () => particles.start(), () => particles.stop());

  onSlide('stats', (slide) => runCounters(slide));

  const ring = initRing($('.ring-stage'));
  onSlide('work', () => ring.start(), () => ring.stop());

  if (finePointer) {
    initCursor();
    initMagnetic();
    initTilt();
    if (!reduceMotion) initParallax();
  }

  const fromHash = slides.findIndex((s) => `#${s.id}` === location.hash);
  runPreloader().then(() => goTo(Math.max(0, fromHash), { instant: true }));
})();
