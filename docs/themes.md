# Themes

Nenpi defaults to Codex-style colors: terminal foreground/background, cyan
accents, and subdued labels. The terminal and browser UI use the same theme.
For a light terminal, select `codex-light` (Codex's `#005f87` accent).
Set the theme in `~/.config/nenpi/config.toml`:

```toml
[theme]
name = "codex" # or "codex-light"

[theme.colors] # optional overrides
accent = "cyan"
primary = "cyan"
codex = "cyan" # CLI Codex bars
claude = "default" # CLI Claude bars
# background = "#181818"
# foreground = "#eeeeee"
```

Colors accept `default`, `black`, `red`, `green`, `yellow`, `blue`, `magenta`,
`cyan`, `white`, or `#RRGGBB`. UI roles are `primary`, `accent`, `foreground`,
`background`, `surface`, `panel`, `success`, `warning`, and `error`; CLI roles
include `claude` and `codex`. Diagnostic reports (`auto` and `tools --explain`)
use these text classifications:

| Role | Meaning | Default appearance |
| --- | --- | --- |
| `heading` | Report and section titles | Bold |
| `label` | Field names and row identities | Terminal foreground |
| `data` | Individual measurements | Terminal foreground |
| `summary` | A finding's interpretation | Bold |
| `total` | Combined quantities and shares | Bold |
| `instruction` | Commands and next actions | Cyan |
| `detail` | Scope, provenance, estimates and caveats | Dim |
| `warning` | A pattern or limitation worth inspecting | Yellow |
| `improvement` | A reduction or positive modeled savings | Green |
| `cost` | Negative modeled savings | Red |
| `table_header` | Table column names | Bold syntax-theme accent |
| `command` | Shell executable | Syntax-theme blue |
| `command_option` | Option name | Syntax-theme red |
| `command_punctuation` | Option dashes | Syntax-theme muted foreground |
| `command_argument` | Unquoted argument | Syntax-theme foreground |
| `command_string` | Quoted argument | Syntax-theme green |

Each span has a role; mixed lines can contain several roles. A Shake context
reduction is an `improvement`, while an increase in uncached input is a
`warning`. Neither claims net quota savings. Ordinary measurements are `data`,
not warnings merely because their values are large.

Override any role under `[theme.colors]`, for example `total = "magenta"`,
`label = "blue"`, or `instruction = "#00aaff"`. Renderers contain no palette
choices. Older `replay`, `repeat`, `output`, and `next` keys remain accepted for
compatibility but are no longer used by these diagnostic reports.

To inspect the output in your terminal now:

```sh
nenpi auto --color always
nenpi tools --session SESSION --prompt 2 --explain --color always
nenpi auto --no-color
```

Explicit `--color always` overrides `NO_COLOR`; `--no-color` takes precedence.
Labels, measurements and layout remain the same without color.

 `NENPI_THEME=codex-light` overrides the
configured name for a run, including `nenpi-web`. Restart the UI after editing
colors. Source changes preserve theme settings. CLI `--no-color` and `NO_COLOR`
suppress report colors; redirected reports stay plain text. Browser colors use
the browser terminal palette, so they may differ from your terminal's palette.

[Basic usage](../README.md)
