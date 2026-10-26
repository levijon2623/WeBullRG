"""
llm_service.py
==============
The Claude jobs, as one service (cleanbot-llm, deploy/llm.service), in its own venv:
  * every 60 s: score the relevant headlines news_logger.py has logged since the last pass
    (news_score.py) -> live/news/ai/DATE.jsonl
  * weekdays from 09:05 ET: the pre-open brief (morning_brief.py) -> live/brief/DATE.json / .md,
    once the pre-open positioning build exists; retried every 5 min until 09:25 if a call fails
Starting, stopping or crashing it never touches the bot, the viewer or the other services.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import llm_common as L
import morning_brief
import news_score

BRIEF_START, BRIEF_END = 9 * 60 + 5, 9 * 60 + 25


def main():
    print("  🤖 [LLM] service up (headline scoring every 60 s; pre-open brief from 09:05 ET)", flush=True)
    said, last_brief_try = set(), 0.0
    while True:
        now = dt.datetime.now(L.NY)
        day, mod = now.date(), now.hour * 60 + now.minute
        try:
            n, err = news_score.score_pass(day)
            if n:
                print(f"  🤖 [LLM] scored {n} headline(s)", flush=True)
            if err and (day, err) not in said:
                said.add((day, err))
                print(f"  🤖 [LLM] scoring: {err}", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- keep the service alive
            print(f"  🤖 [LLM] scoring error {type(e).__name__}: {str(e)[:160]}", flush=True)
        brief_p = os.path.join(morning_brief.LIVE, "brief", f"{day}.json")
        if (now.weekday() < 5 and BRIEF_START <= mod <= BRIEF_END and not os.path.exists(brief_p)
                and morning_brief.built_pre(day) and time.time() - last_brief_try >= 300):
            last_brief_try = time.time()
            try:
                g, _ = morning_brief.run(day, now)
                print(f"  🤖 [LLM] pre-open brief written ({len(g['flags'])} flags)", flush=True)
            except Exception as e:                                   # noqa: BLE001
                print(f"  🤖 [LLM] brief error {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
