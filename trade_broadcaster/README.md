# trade_broadcaster

Posts your trades to Telegram and/or Discord as they are **placed** and **filled**,
read straight from your brokerage account. Read-only: it never places, changes or
cancels an order.

```
🟢 OPENED · BUY TO OPEN SPY 772C 0DTE
2 contracts · @ $0.92
10:41:07 ET · @yourhandle

🔴 CLOSED · SELL TO CLOSE SPY 772C 0DTE
2 contracts · @ $1.20
📈 +30.4% on the position
10:58:31 ET · @yourhandle
```

## Quick start

```bash
cd trade_broadcaster/..            # the folder that CONTAINS trade_broadcaster/
pip install -r trade_broadcaster/requirements.txt
cp trade_broadcaster/config.example.toml config.toml
cp trade_broadcaster/.env.example .env      # fill in your secrets
python -m trade_broadcaster login           # once: Robinhood code / app approval
python -m trade_broadcaster check           # what it sees -- posts nothing
python -m trade_broadcaster test-sinks      # one test post to each destination
python -m trade_broadcaster run             # go
```

The first `run` only **records** the orders already in the account. It posts
what happens from then on, so starting it never floods a channel with history.

## Destinations

* **Telegram**: create a bot with @BotFather (token), add it to your channel as an
  admin, and set the chat id (`@channelname` or the numeric `-100…` id).
* **Discord**: Channel settings → Integrations → Webhooks → New webhook → copy URL.

The token and webhook URL *are* the credentials. Keep them in `.env` only; the
broadcaster never logs them.

## Robinhood login

Robinhood has **no official API** for stocks/options. This uses the community
library `robin_stocks`, which drives the same private endpoints as the app.
Robinhood can change those without notice, and automated access may conflict with
its customer agreement. This tool only reads one page of recent orders every few
seconds. Use it on your own account at your own discretion.

* Run `python -m trade_broadcaster login` once in a terminal. Approve in the app or
  enter the SMS code. The session is saved in `state/` and reused.
* For unattended re-logins, set up an **authenticator app** for Robinhood and put
  its base32 secret in `RH_TOTP_SECRET`. Without it, a background run that needs a
  new code stops and says so, rather than hanging.

## Privacy & timing (`[privacy]`)

`show_quantity`, `show_price`, `show_pnl`, `delay_seconds`, which events to `post`,
and a `footer` line. % P&L is shown on a closing fill only when the broadcaster also
saw the opening fill, so it never guesses a cost.

## Running it all day (Linux, systemd)

```ini
# /etc/systemd/system/trade-broadcaster.service
[Unit]
Description=trade_broadcaster
After=network-online.target
[Service]
WorkingDirectory=/path/to/folder-containing-trade_broadcaster
ExecStart=/usr/bin/python3 -m trade_broadcaster run
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
```

## Adding a broker (a source)

Create `trade_broadcaster/sources/mybroker.py`:

```python
from ..registry import register_source
from ..events import OrderSnapshot, Leg, Instrument, OPEN, FILLED
from .base import SnapshotSource, SourceError

@register_source("mybroker")
class MyBroker(SnapshotSource):
    def start(self):
        ...                                   # log in; raise SourceError on failure
    def fetch_snapshots(self):
        return [OrderSnapshot(...) for o in recent_orders()]   # orders as they are NOW
```

That's all: the shared `OrderTracker` turns snapshots into placed / partial /
filled / cancelled posts, handles the first-run baseline and the P&L, and the
outbox handles delays, retries and de-duplication. Feeds that are already events
(a ledger file, a webhook inbox) subclass `Source` and return `TradeEvent`s from
`poll()`. See `sources/jsonl_ledger.py`.

## Adding a destination (a sink)

`trade_broadcaster/sinks/mysink.py` with `@register_sink("mysink")` and a
`send(message)` that raises `SinkError` on failure (set `retry_after` for rate
limits). `message.title`, `message.lines`, `message.footer` and `message.color`
are provided.

## A note on posting to an audience

Posting your own trades to your own or a friend's channel is one thing. Charging
for a trade feed, or presenting performance to attract subscribers, can bring
securities rules into play. Talk to a lawyer before going there.

## Tests

`python -m unittest discover -s trade_broadcaster/tests -t .`
