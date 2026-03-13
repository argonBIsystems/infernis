# INFERNIS Brand Guidelines

## Logo

The INFERNIS logo is a **monospaced wordmark** — `INFERNIS` in JetBrains Mono with the `I` in ember red. That's it. No icons, no gradients, no clip-art flames.

### Files

| File | Use |
|------|-----|
| `infernis-logo-dark.svg` | **Primary** — white text on dark backgrounds |
| `infernis-logo-light.svg` | Dark text on light backgrounds |
| `infernis-icon.svg` | The ember `I` on charcoal — favicons, avatars, small sizes |

### Construction

```
INFERN I S
       ^
       ember red (#E8531E)
```

- Font: JetBrains Mono, weight 500
- Letter spacing: 0.15em
- The `I` is the only character in color — everything else is neutral

## Colors

### Core Palette

| Name | Hex | CSS Variable | Usage |
|------|-----|-------------|-------|
| **Ember** | `#E8531E` | `--ember` | Primary brand color, CTAs, the `I`, section labels |
| Ember Light | `#F4845F` | `--ember-light` | Links, hover states, secondary accents |
| Ember Glow | `#FF6B35` | `--ember-glow` | Hover effects, gradient endpoints |
| Charcoal | `#0D0D0D` | `--charcoal` | Page background |
| Ash | `#1A1A1A` | `--ash` | Card/section background |
| Smoke | `#2A2A2A` | `--smoke` | Borders, dividers |
| Fog | `#8A8A8A` | `--fog` | Secondary text, muted content |
| Cloud | `#B0B0B0` | `--cloud` | Body text |
| Snow | `#F0EDE8` | `--snow` | Warm white, subtle backgrounds |
| White | `#FAFAF8` | `--white` | Headings, high-emphasis text |
| Forest | `#1B4332` | `--forest` | Open-source/free tier accent |
| Forest Light | `#2D6A4F` | `--forest-light` | Free tier borders, green badges |

### Danger Level Colors

Canonical colors for API responses and all visualizations:

| Level | Hex | CSS |
|-------|-----|-----|
| VERY_LOW | `#22C55E` | Green |
| LOW | `#3B82F6` | Blue |
| MODERATE | `#EAB308` | Yellow |
| HIGH | `#F97316` | Orange |
| VERY_HIGH | `#EF4444` | Red |
| EXTREME | `#1A0000` | Near-black red |

## Typography

| Role | Font | Weight | Usage |
|------|------|--------|-------|
| Headings | Playfair Display | 700–900 | Section titles, hero text |
| Body | DM Sans | 300–600 | Paragraphs, UI text |
| Mono/Labels | JetBrains Mono | 400–500 | Logo, section labels, code, stats, nav |

Google Fonts import:
```html
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;800;900&display=swap" rel="stylesheet">
```

## Design Principles

- **Dark-first** — charcoal background is the default. The site is dark.
- **Editorial tone** — Playfair Display serif headings give it weight. This isn't a playful startup.
- **Ember as accent, not flood** — the red is used sparingly: the `I`, section labels, CTAs, hover states. Most of the page is neutral.
- **Monospace for authority** — stats, labels, and technical content use JetBrains Mono.
- **No decorative elements** — no icons, no illustrations, no stock photos. The data speaks.

## Usage

### Do
- Use the wordmark with the ember `I` exactly as provided
- Maintain clear space around the logo (minimum: 1x the height of the `I`)
- Use on dark backgrounds (charcoal or ash) for primary placement

### Don't
- Add flame icons or fire imagery to the logo
- Change the ember color or which letter is highlighted
- Use the logo on busy or colorful backgrounds
- Stretch, rotate, or add effects

## Attribution

When using INFERNIS data or API:

```
Wildfire risk data by INFERNIS (infernis.ca)
```

## License

Brand assets in this directory are provided for:
- Contributing to the INFERNIS project
- Referencing INFERNIS in documentation, presentations, or blog posts
- Building integrations with the INFERNIS API

Do not use to imply endorsement of unrelated products.
