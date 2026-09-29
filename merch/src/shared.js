// Общие SVG-определения для всех макетов: фильтры, сова-маскот, эмблема, иконки, стикеры.
// Подключается первым скриптом в <body> каждой сцены.

function starburst(cx, cy, rOut, rIn, n) {
  let d = '';
  for (let i = 0; i < n * 2; i++) {
    const r = i % 2 ? rIn : rOut;
    const a = (Math.PI * i) / n - Math.PI / 2;
    d += (i ? 'L' : 'M') + (cx + r * Math.cos(a)).toFixed(1) + ' ' + (cy + r * Math.sin(a)).toFixed(1);
  }
  return d + 'Z';
}

const HEART = 'M12 21.3 L10.5 20 C5.4 15.4 2 12.3 2 8.5 C2 5.4 4.4 3 7.5 3 C9.2 3 10.9 3.8 12 5.1 C13.1 3.8 14.8 3 16.5 3 C19.6 3 22 5.4 22 8.5 C22 12.3 18.6 15.4 13.5 20 Z';
const SPARKLE = 'M12 0 C12.9 7.2 16.8 11.1 24 12 C16.8 12.9 12.9 16.8 12 24 C11.1 16.8 7.2 12.9 0 12 C7.2 11.1 11.1 7.2 12 0 Z';
const BOLT = 'M13.5 1.5 L4 13.5 H11 L9.5 22.5 L20 9.5 H13 Z';

const DEFS = `
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
<defs>
  <!-- ===== Фильтры ===== -->
  <filter id="grain" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="4" stitchTiles="stitch"/>
    <feColorMatrix type="saturate" values="0"/>
  </filter>

  <filter id="knit" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="1.1 0.45" numOctaves="3" seed="7"/>
    <feColorMatrix type="saturate" values="0"/>
  </filter>

  <filter id="canvas" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="turbulence" baseFrequency="0.7 0.7" numOctaves="2" seed="11"/>
    <feColorMatrix type="saturate" values="0"/>
  </filter>

  <filter id="paper" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="4" seed="9"/>
    <feColorMatrix type="saturate" values="0"/>
  </filter>

  <filter id="b2"><feGaussianBlur stdDeviation="2"/></filter>
  <filter id="b4" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4"/></filter>
  <filter id="b8" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="8"/></filter>
  <filter id="b14" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="14"/></filter>
  <filter id="b24" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="24"/></filter>
  <filter id="b40" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="40"/></filter>

  <!-- Шелкография: краска чуть «садится» в ткань -->
  <filter id="print" x="-5%" y="-5%" width="110%" height="110%">
    <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3" result="t"/>
    <feDisplacementMap in="SourceGraphic" in2="t" scale="1.4" xChannelSelector="R" yChannelSelector="G" result="d"/>
    <feComponentTransfer in="d"><feFuncA type="linear" slope="0.94"/></feComponentTransfer>
  </filter>

  <!-- Вышивка: лёгкий объём -->
  <filter id="emb" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="1.6" stdDeviation="1" flood-color="#0b0420" flood-opacity=".6"/>
  </filter>

  <!-- Шениль (махровые нашивки варсити) -->
  <filter id="chenille" x="-10%" y="-10%" width="120%" height="120%">
    <feTurbulence type="fractalNoise" baseFrequency="1.3" numOctaves="2" seed="5" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="3.5" xChannelSelector="R" yChannelSelector="G" result="d"/>
    <feColorMatrix in="n" type="matrix" values=".33 .33 .33 0 0  .33 .33 .33 0 0  .33 .33 .33 0 0  0 0 0 0 1" result="g"/>
    <feComponentTransfer in="g" result="g2">
      <feFuncR type="linear" slope=".6" intercept=".6"/><feFuncG type="linear" slope=".6" intercept=".6"/><feFuncB type="linear" slope=".6" intercept=".6"/>
    </feComponentTransfer>
    <feBlend in="d" in2="g2" mode="multiply" result="m"/>
    <feComposite in="m" in2="d" operator="in" result="mc"/>
    <feDropShadow in="mc" dx="0" dy="3" stdDeviation="2.5" flood-color="#0b0420" flood-opacity=".55"/>
  </filter>

  <!-- Фетровая подложка нашивки -->
  <filter id="felt" x="-15%" y="-15%" width="130%" height="130%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="6" result="d"/>
    <feFlood flood-color="#ffffff"/><feComposite in2="d" operator="in" result="white"/>
    <feMerge><feMergeNode in="white"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>

  <!-- Высечка стикера: белая кромка + тень -->
  <filter id="diecut" x="-25%" y="-25%" width="150%" height="150%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="7" result="d"/>
    <feFlood flood-color="#ffffff"/><feComposite in2="d" operator="in" result="white"/>
    <feGaussianBlur in="d" stdDeviation="5" result="b"/><feOffset in="b" dx="2" dy="7" result="bo"/>
    <feFlood flood-color="#1c0a3f" flood-opacity=".32"/><feComposite in2="bo" operator="in" result="shadow"/>
    <feMerge><feMergeNode in="shadow"/><feMergeNode in="white"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>

  <filter id="drop" x="-25%" y="-25%" width="150%" height="150%">
    <feDropShadow dx="3" dy="9" stdDeviation="7" flood-color="#1c0a3f" flood-opacity=".38"/>
  </filter>

  <!-- ===== Градиенты ===== -->
  <linearGradient id="silver" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset=".4" stop-color="#b9b6c6"/>
    <stop offset=".65" stop-color="#f4f2fa"/><stop offset="1" stop-color="#8d89a0"/>
  </linearGradient>
  <radialGradient id="gloss" cx=".35" cy=".3" r=".8">
    <stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/>
  </radialGradient>

  <!-- ===== Сова-маскот «Совёнок ИУЭС» ===== -->
  <symbol id="owl" viewBox="0 0 200 244">
    <path d="M36 104 L14 62 L66 82 Z" fill="var(--owl-body)"/>
    <path d="M164 104 L186 62 L134 82 Z" fill="var(--owl-body)"/>
    <path d="M100 56 C152 56 178 96 178 146 C178 200 144 232 100 232 C56 232 22 200 22 146 C22 96 48 56 100 56 Z" fill="var(--owl-body)"/>
    <path d="M27 140 C12 170 22 206 54 216 C43 192 40 166 27 140 Z" fill="var(--owl-ink)" opacity=".28"/>
    <path d="M173 140 C188 170 178 206 146 216 C157 192 160 166 173 140 Z" fill="var(--owl-ink)" opacity=".28"/>
    <path d="M100 152 C132 152 148 176 148 198 C148 219 127 229 100 229 C73 229 52 219 52 198 C52 176 68 152 100 152 Z" fill="var(--owl-belly)"/>
    <g fill="none" stroke="var(--owl-body)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M80 180 l7 7 l7 -7"/><path d="M106 180 l7 7 l7 -7"/>
      <path d="M67 202 l7 7 l7 -7"/><path d="M93 202 l7 7 l7 -7"/><path d="M119 202 l7 7 l7 -7"/>
    </g>
    <circle cx="72" cy="120" r="37" fill="var(--owl-face)"/>
    <circle cx="128" cy="120" r="37" fill="var(--owl-face)"/>
    <circle cx="72" cy="120" r="24" fill="#fff"/>
    <circle cx="128" cy="120" r="24" fill="#fff"/>
    <circle cx="76" cy="123" r="13" fill="var(--owl-ink)"/>
    <circle cx="124" cy="123" r="13" fill="var(--owl-ink)"/>
    <circle cx="80.5" cy="117" r="4.6" fill="#fff"/>
    <circle cx="128.5" cy="117" r="4.6" fill="#fff"/>
    <g fill="none" stroke="var(--owl-ink)" stroke-width="6">
      <circle cx="72" cy="120" r="29"/><circle cx="128" cy="120" r="29"/>
    </g>
    <path d="M89 143 L111 143 L100 161 Z" fill="var(--owl-accent)" stroke="var(--owl-ink)" stroke-width="3.5" stroke-linejoin="round"/>
    <g fill="var(--owl-accent)" stroke="var(--owl-ink)" stroke-width="3">
      <ellipse cx="82" cy="233" rx="14" ry="8"/><ellipse cx="118" cy="233" rx="14" ry="8"/>
    </g>
    <path d="M60 58 L60 82 Q100 100 140 82 L140 58 Z" fill="var(--owl-ink)"/>
    <path d="M100 16 L178 46 L100 76 L22 46 Z" fill="var(--owl-ink)"/>
    <path d="M100 22 L166 46 L100 70 L34 46 Z" fill="#fff" opacity=".07"/>
    <path d="M100 46 Q140 50 162 56 L163 94" fill="none" stroke="var(--owl-accent)" stroke-width="4.5" stroke-linecap="round"/>
    <path d="M157 90 L169 90 L173 112 L153 112 Z" fill="var(--owl-accent)"/>
    <circle cx="100" cy="46" r="6.5" fill="var(--owl-accent)"/>
  </symbol>

  <!-- ===== Иконки 24×24 (цвет = currentColor) ===== -->
  <symbol id="i-econ" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M5 21v-5M11 21v-8M17 21v-11"/><path d="M3 11 L9 6 L13 9 L20.5 2.5"/><path d="M15.5 2.5h5v5"/>
  </symbol>
  <symbol id="i-eco" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 20 C4 10 10 4 20.5 3.5 C20 14 14 20 4 20 Z"/><path d="M4 20 L14 10"/>
  </symbol>
  <symbol id="i-soc" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="8" cy="8" r="3.4"/><circle cx="17" cy="9" r="2.8"/>
    <path d="M2 21c0-3.8 2.7-6.4 6-6.4s6 2.6 6 6.4"/><path d="M15 14.6c.6-.2 1.3-.3 2-.3 3 0 5 2.2 5 5.7"/>
  </symbol>
  <symbol id="i-coffee" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3.5 9.5h13v5.5a5.5 5.5 0 0 1 -5.5 5.5h-2a5.5 5.5 0 0 1 -5.5 -5.5 Z"/><path d="M16.5 11h1.5a2.6 2.6 0 0 1 0 5.2h-1.8"/>
    <path d="M8 2.5c1.4 1.4 -1.4 2.6 0 4M12 2.5c1.4 1.4 -1.4 2.6 0 4"/>
  </symbol>
  <symbol id="i-plane" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 2 L11 13"/><path d="M22 2 L15 22 L11 13 L2 9 Z"/>
  </symbol>
  <symbol id="i-cap" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 4 L23 9 L12 14 L1 9 Z"/><path d="M6 11.5 V16 c0 1.8 2.7 3 6 3 s6 -1.2 6 -3 V11.5"/><path d="M21 9.8 V15"/>
  </symbol>
  <symbol id="i-lighthouse" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 21 L10.4 9 H13.6 L15 21 Z"/><path d="M9.6 9 h4.8 M10.5 9 V6 h3 V9 M12 6 V4"/>
    <path d="M16 5.5 l4 -1.6 M16 8 l4 1.2 M8 5.5 l-4 -1.6 M8 8 l-4 1.2"/><path d="M4 21 h16"/>
  </symbol>
  <symbol id="i-wave" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
    <path d="M1 10 c2.5 0 2.5 -2.5 5 -2.5 s2.5 2.5 5 2.5 2.5 -2.5 5 -2.5 2.5 2.5 5 2.5"/>
    <path d="M1 16 c2.5 0 2.5 -2.5 5 -2.5 s2.5 2.5 5 2.5 2.5 -2.5 5 -2.5 2.5 2.5 5 2.5"/>
  </symbol>
  <symbol id="i-bolt" viewBox="0 0 24 24"><path d="${BOLT}" fill="currentColor"/></symbol>
  <symbol id="i-star" viewBox="0 0 24 24"><path d="${SPARKLE}" fill="currentColor"/></symbol>
  <symbol id="i-heart" viewBox="0 0 24 24"><path d="${HEART}" fill="currentColor"/></symbol>

  <!-- ===== Эмблема-печать ===== -->
  <symbol id="emblem" viewBox="0 0 300 300">
    <circle cx="150" cy="150" r="147" fill="var(--em-bg, #3b1886)"/>
    <circle cx="150" cy="150" r="137" fill="none" stroke="var(--em-fg, #d2ff3f)" stroke-width="3.5"/>
    <circle cx="150" cy="150" r="92" fill="var(--em-inner, #5b2bd0)" stroke="var(--em-fg, #d2ff3f)" stroke-width="3.5"/>
    <path id="em-arc" d="M150 150 m-106 0 a106 106 0 1 1 212 0 a106 106 0 1 1 -212 0" fill="none"/>
    <text font-family="Unbounded" font-weight="800" font-size="24" fill="var(--em-fg, #d2ff3f)">
      <textPath href="#em-arc" textLength="662" lengthAdjust="spacing">ИУЭС • ЮФУ • ТАГАНРОГ • СТУДЕНТ •</textPath>
    </text>
    <use href="#owl" x="96" y="84" width="108" height="132"/>
  </symbol>

  <!-- ===== Словесный знак ===== -->
  <symbol id="wordmark" viewBox="0 0 440 150">
    <text x="0" y="112" font-family="Unbounded" font-weight="900" font-size="100" fill="var(--wm, #d2ff3f)">ИУЭС</text>
    <path d="${SPARKLE}" transform="translate(386 6) scale(2.1)" fill="var(--wm-star, var(--wm, #d2ff3f))"/>
    <text x="3" y="146" font-family="Manrope" font-weight="800" font-size="22" letter-spacing="7.3" fill="var(--wm-sub, var(--wm, #d2ff3f))">ЮФУ • ТАГАНРОГ</text>
  </symbol>

  <!-- ===== Стикеры ===== -->
  <symbol id="st-zachet" viewBox="0 0 260 150">
    <rect x="6" y="6" width="248" height="138" rx="24" fill="#7b4dff"/>
    <rect x="18" y="18" width="224" height="114" rx="15" fill="none" stroke="#d2ff3f" stroke-width="4" stroke-dasharray="11 8"/>
    <text x="130" y="88" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="42" fill="#d2ff3f">ЗАЧЁТ</text>
    <text x="130" y="116" text-anchor="middle" font-family="Manrope" font-weight="800" font-size="15" letter-spacing="3" fill="#fff">АВТОМАТОМ ✓</text>
  </symbol>

  <symbol id="st-deadline" viewBox="0 0 260 190">
    <path d="M30 8 H230 a22 22 0 0 1 22 22 V118 a22 22 0 0 1 -22 22 H112 L68 182 L78 140 H30 a22 22 0 0 1 -22 -22 V30 a22 22 0 0 1 22 -22 Z" fill="#d2ff3f"/>
    <text x="130" y="66" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="23" fill="#1c0a3f">ДЕДЛАЙН?</text>
    <text x="130" y="106" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="21" fill="#5b2bd0">УПРАВИМСЯ!</text>
  </symbol>

  <symbol id="st-coffee" viewBox="0 0 220 250">
    <path d="M92 38 c-9 -9 9 -15 0 -26 M128 38 c-9 -9 9 -15 0 -26" fill="none" stroke="#d2ff3f" stroke-width="7" stroke-linecap="round"/>
    <path d="M52 70 H168 L154 228 a10 10 0 0 1 -10 9 H76 a10 10 0 0 1 -10 -9 Z" fill="#7b4dff"/>
    <path d="M60 46 H160 L168 58 H52 Z" fill="#e4d9ff"/>
    <rect x="42" y="56" width="136" height="22" rx="8" fill="#b8a1ff"/>
    <path d="M56 116 H164 L158.5 178 H61.5 Z" fill="#d2ff3f"/>
    <text x="110" y="157" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="25" fill="#1c0a3f">КОФЕ</text>
    <text x="110" y="208" text-anchor="middle" font-family="Manrope" font-weight="800" font-size="13" letter-spacing="1.5" fill="#fff">ТОПЛИВО</text>
    <text x="110" y="224" text-anchor="middle" font-family="Manrope" font-weight="800" font-size="13" letter-spacing="1.5" fill="#fff">СЕССИИ</text>
  </symbol>

  <symbol id="st-lighthouse" viewBox="0 0 220 220">
    <clipPath id="lh-clip"><circle cx="110" cy="110" r="104"/></clipPath>
    <circle cx="110" cy="110" r="104" fill="#3b1886"/>
    <g clip-path="url(#lh-clip)">
      <path d="M110 70 L-10 30 L-10 100 Z" fill="#d2ff3f" opacity=".22"/>
      <path d="M110 70 L230 30 L230 100 Z" fill="#d2ff3f" opacity=".22"/>
      <circle cx="160" cy="52" r="13" fill="#d2ff3f"/>
      <path d="M96 156 L101 82 H119 L124 156 Z" fill="#fff"/>
      <path d="M99.3 102 H120.7 L121.6 116 H98.4 Z M97.5 132 H122.5 L123.3 144 H96.7 Z" fill="#7b4dff"/>
      <rect x="96" y="77" width="28" height="6" rx="2" fill="#fff"/>
      <rect x="101" y="63" width="18" height="14" fill="#d2ff3f"/>
      <path d="M97 64 L110 50 L123 64 Z" fill="#fff"/>
      <path d="M66 162 Q110 140 154 162 Z" fill="#1c0a3f"/>
      <path d="M-10 160 Q10 150 30 160 T70 160 T110 160 T150 160 T190 160 T230 160 V240 H-10 Z" fill="#5b2bd0"/>
      <path d="M-10 176 Q10 166 30 176 T70 176 T110 176 T150 176 T190 176 T230 176 V240 H-10 Z" fill="#7b4dff"/>
    </g>
    <text x="110" y="200" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="15" fill="#fff">ТАГАНРОГ</text>
  </symbol>

  <symbol id="st-heart" viewBox="0 0 240 220">
    <path d="${HEART}" transform="translate(12 4) scale(9)" fill="#ff77c8"/>
    <text x="120" y="92" text-anchor="middle" font-family="Manrope" font-weight="800" font-size="16" letter-spacing="2" fill="#fff">Я ЛЮБЛЮ</text>
    <text x="120" y="130" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="31" fill="#fff">ИУЭС</text>
  </symbol>

  <symbol id="st-cap" viewBox="0 0 240 200">
    <path d="M58 82 V122 C58 146 182 146 182 122 V82 L120 106 Z" fill="#1c0a3f"/>
    <path d="M120 18 L228 60 L120 102 L12 60 Z" fill="#1c0a3f"/>
    <path d="M120 26 L212 60 L120 94 L28 60 Z" fill="#fff" opacity=".08"/>
    <path d="M120 60 L200 76 V118" fill="none" stroke="#d2ff3f" stroke-width="5" stroke-linecap="round"/>
    <path d="M194 114 h12 l4 24 h-20 Z" fill="#d2ff3f"/>
    <circle cx="120" cy="60" r="7" fill="#d2ff3f"/>
    <path d="M6 148 H234 L222 169 L234 190 H6 L18 169 Z" fill="#d2ff3f"/>
    <text x="120" y="177" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="19" fill="#1c0a3f">ВЫПУСК 2030</text>
  </symbol>

  <symbol id="st-five" viewBox="0 0 200 200">
    <path d="${starburst(100, 100, 96, 80, 16)}" fill="#d2ff3f"/>
    <circle cx="100" cy="100" r="70" fill="none" stroke="#1c0a3f" stroke-width="3" stroke-dasharray="6 6"/>
    <text x="100" y="128" text-anchor="middle" font-family="Unbounded" font-weight="900" font-size="88" fill="#5b2bd0">5</text>
    <text x="100" y="156" text-anchor="middle" font-family="Manrope" font-weight="800" font-size="15" letter-spacing="3" fill="#1c0a3f">ОТЛИЧНО</text>
  </symbol>

  <symbol id="st-owl" viewBox="0 0 240 290">
    <use href="#owl" x="20" y="0" width="200" height="244"/>
    <path d="M0 238 H240 L228 260 L240 282 H0 L12 260 Z" fill="#d2ff3f"/>
    <text x="120" y="267" text-anchor="middle" font-family="Unbounded" font-weight="800" font-size="15" fill="#1c0a3f">НЕ СПЛЮ • УЧУСЬ</text>
  </symbol>

  <symbol id="st-tri" viewBox="0 0 300 140">
    <rect x="4" y="4" width="292" height="132" rx="30" fill="#3b1886"/>
    <circle cx="80" cy="54" r="30" fill="#d2ff3f"/><use href="#i-econ" x="62" y="36" width="36" height="36" color="#1c0a3f"/>
    <circle cx="150" cy="54" r="30" fill="#b8a1ff"/><use href="#i-eco" x="132" y="36" width="36" height="36" color="#1c0a3f"/>
    <circle cx="220" cy="54" r="30" fill="#ff77c8"/><use href="#i-soc" x="202" y="36" width="36" height="36" color="#1c0a3f"/>
    <text x="150" y="116" text-anchor="middle" font-family="Unbounded" font-weight="800" font-size="14" fill="#fff">ЭКО • ЭКОНОМ • СОЦИУМ</text>
  </symbol>

  <symbol id="st-bolt" viewBox="0 0 24 24">
    <path d="${BOLT}" fill="#d2ff3f" stroke="#1c0a3f" stroke-width="1.2" stroke-linejoin="round"/>
  </symbol>
  <symbol id="st-star" viewBox="0 0 24 24"><path d="${SPARKLE}" fill="#b8a1ff"/></symbol>

  <symbol id="st-wm" viewBox="0 0 320 130">
    <rect x="4" y="4" width="312" height="122" rx="61" fill="#7b4dff"/>
    <text x="42" y="84" font-family="Unbounded" font-weight="900" font-size="56" fill="#d2ff3f">ИУЭС</text>
    <path d="${SPARKLE}" transform="translate(262 44) scale(1.7)" fill="#fff"/>
  </symbol>
</defs>
</svg>`;

document.body.insertAdjacentHTML('afterbegin', DEFS);
