import os
import json
import threading
from datetime import datetime

try:
    import anthropic
except ImportError:
    anthropic = None

MODEL = "claude-haiku-5-5"      # fast and cheap; marchukovachu


class TrafficReportGenerator:

    def __init__(self, use_llm=True):
        self.client = None
        if use_llm and anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic()

        self.mode = "LLM" if self.client else "TEMPLATE"
        self.latest = ""
        self._busy = False
        self._seq = 0
        self._warned = False

        print(f"[GenAI] mode: {self.mode}")
        if self.mode == "TEMPLATE":
                        print("[GenAI] API key or anthropic package not found -> using template text")

    # ---------------- LLM call ----------------

    def _ask(self, prompt, max_tokens=300):
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()

    # ---------------- per-decision explanation ----------------

    @staticmethod
    def _template(counts, selected, decisions, ambulance_routes, reason):
        d = decisions[selected]
        if selected in ambulance_routes:
            return (
                f"Ambulance on {selected}: emergency preemption, "
                f"GREEN for {d['green_time']} s."
            )
        return (
            f"{selected} gets GREEN for {d['green_time']} s "
            f"({counts[selected]} vehicles, {d['density']}): {reason}."
        )

    def submit(self, video_time, counts, selected, decisions,
               ambulance_routes, reason=""):
        # Template text ventane (screen lo aagakunda)
        self.latest = self._template(
            counts, selected, decisions, ambulance_routes, reason
        )

        if self.client is None or self._busy:
            return

        self._seq += 1
        seq = self._seq
        d = decisions[selected]

        prompt = (
            "You are the explanation module of a smart traffic signal system.\n"
            "System rules: if an ambulance is confirmed on a route, that route gets "
            "GREEN first (preemption, planned 40s, held while the ambulance is "
            "present, max 60s). Otherwise every route gets one GREEN per cycle, "
            "densest route first. Green time by vehicle count: <=10: 10s, 11-15: 20s, "
            "16-20: 30s, 21-25: 40s, 26-30: 45s, >30: 50s. Each GREEN is followed by "
            "3s yellow and 1s all-red.\n"
            f"Facts: vehicle counts per route = {counts}; "
            f"ambulance routes = {list(ambulance_routes)}; selected = {selected}; "
            f"green time = {d['green_time']} s; density = {d['density']}; "
            f"reason = {reason}.\n"
            "Write ONE plain-ASCII sentence (max 25 words) explaining this decision. "
            "Use only the facts above."
        )

        def worker():
            self._busy = True
            try:
                text = self._ask(prompt, max_tokens=80)
                if seq == self._seq:          # pata result kotha daanni overwrite cheyyakunda
                    self.latest = text
            except Exception as e:
                if not self._warned:
                    print("[GenAI] LLM error:", e)
                    self._warned = True
            finally:
                self._busy = False

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- final report ----------------

    def save_report(self, decision_log, peak_counts, green_log, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        routes = list(peak_counts.keys())
        n = len(decision_log)

        avg = {
            r: round(sum(x["counts"][r] for x in decision_log) / n, 1) if n else 0
            for r in routes
        }
        amb_events = [
            {"time_s": x["time"], "routes": x["ambulance"], "selected": x["selected"]}
            for x in decision_log if x["ambulance"]
        ]
        facts = {
            "total_green_phases": n,
            "avg_vehicles_per_route": avg,
            "peak_vehicles_per_route": peak_counts,
            "times_green_per_route": green_log,
            "max_green_time_s": max((x["green_time"] for x in decision_log), default=0),
            "ambulance_events": amb_events,
        }

        body = None
        if self.client is not None:
            prompt = (
                "Write a short markdown report (about 200 words) for a smart traffic "
                "signal demo (4 routes, vehicle-density based green time, round-robin "
                "cycle, ambulance preemption). Sections: Overview, Traffic "
                "observations, Ambulance events, Conclusion. Use ONLY the facts below "
                "and do not invent any number.\n\nFACTS:\n" + json.dumps(facts, indent=2)
            )
            try:
                body = self._ask(prompt, max_tokens=700)
            except Exception as e:
                print("[GenAI] report LLM error:", e)

        if body is None:
            lines = [
                "## Overview",
                f"The system ran {n} green phases across {len(routes)} routes.",
                "",
                "## Ambulance events",
                f"{len(amb_events)} green phase(s) were ambulance preemptions." if amb_events
                else "No ambulance was detected or simulated.",
            ]
            body = "\n".join(lines)

        table = [
            "| Route | Avg vehicles | Peak vehicles | Times GREEN |",
            "|---|---|---|---|",
        ]
        for r in routes:
            table.append(f"| {r} | {avg[r]} | {peak_counts[r]} | {green_log[r]} |")

        text = (
            "# Traffic Signal Run Report\n\n"
            f"_Generated {datetime.now():%Y-%m-%d %H:%M} | GenAI mode: {self.mode}_\n\n"
            + body
            + "\n\n## Data (computed by the system)\n\n"
            + "\n".join(table)
            + "\n"
        )

        path = os.path.join(out_dir, "traffic_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path