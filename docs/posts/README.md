# Share cards

One-image summaries of something the research found, sized for a community
post (Webull, X, Discord). Each card is a single self-contained HTML page that
renders to a PNG with the machine's own Edge or Chrome:

```
python docs/posts/render_post.py docs/posts/<card>.html          # -> <card>.png, 2160x2700
python docs/posts/render_post.py <card>.html --scale 1 --out preview.png   # quick look
```

No packages and no network: fonts are system fonts, and the charts are inline
SVG drawn by the page's own script.

## Shared pieces

- `card.css`: the frame, tokens (validated colors), tiles, trade-list and chip styles.
- `card.js`: `divergingBars` (bought vs sold to open, shared scale across a series of cards),
  `tradeTable` (the "trades that matter" list, with a MON chip for prints Monday's open interest
  confirms) and `compareBars` (one measure across tickers, one hue, no legend needed).
- `make_etf_cards.py`: a worked example of data-driven cards. The numbers live in dicts; re-run it,
  then render. Use it when a series of cards shares one layout (one per ticker).
- A one-off card (e.g. `2026-10-16-etf-overview.html`) links `card.css` / `card.js` and writes its own body.

## Making a new card

1. **Start from the analysis, not the design.** Every number on the card comes
   from a saved tool run (e.g. `probe_uoa.py ... --scan`). Keep the command in the
   commit message, so anyone can reproduce the card.
2. **Copy the latest card** (`2026-10-16-qqq-1030-puts.html`) to
   `YYYY-MM-DD-<ticker>-<topic>.html`. Keep the frame: kicker / headline / one-line
   subhead, a row of 3-4 stat tiles, at most two charts, a "what would change
   the read" box, and the footer.
3. **One idea per card.** The headline states the finding with its number. If it
   needs a third chart, it is two cards.
4. **Color by job.** The palette at the top of the page was validated for the dark
   surface (colorblind separation and contrast) with the dataviz validator:
   blue/red for bought/sold (polarity), violet/aqua for two named series. Reuse
   those roles; validate any new color before using it. Text stays in the ink
   tokens, never in a series color.
5. **Label directly.** Values sit next to their bars and line ends, and a legend is
   present whenever two encodings share a chart. No dual axes.
6. **Footer, always:** the data source, the method or tool, the repo
   (`github.com/levijon2623/WeBullRG`), "analysis with Claude (Anthropic)", and
   "not financial advice".
7. **Render at 1x, look at it, fix collisions, then render at 2x.** The validator
   checks color, not layout.

## Before posting

- Inferences are labelled as inferences ("side read from ask/bid", "if opening").
- Times are ET and dates are explicit (no "yesterday").
- Nothing from the account: no balances, positions, order ids, box addresses or keys.
- Third-party data: the card shows derived figures with attribution. Check the
  provider's terms before posting raw tables.
