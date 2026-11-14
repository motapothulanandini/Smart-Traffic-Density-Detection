class SignalManager:
    """Green / Yellow / All-red state machine.

    Rules:
    1. Ambulance confirm ayithe aa route ki first priority (full green time).
       Vere route green lo unte: yellow, all-red, tarvata ambulance route green.
    2. Ambulance green time aipoyaka: ippudu ekkuva vehicles unna route ki green,
       tarvata migilina routes density order lo (anni routes ki turn).
    3. Normal green time = vehicle count batti (controller tiers).
    """

    YELLOW_SEC = 3.0
    ALL_RED_SEC = 1.0
    WARMUP_SEC = 0.5          # modati decision mundu observe

    AMB_EARLY_END = False     # True ayithe ambulance pothe 10s tarvata aapesthundi
    AMB_MIN_GREEN = 10.0      # (early end True ayithe matrame)
    AMB_CLEAR_SEC = 3.0       # (early end True ayithe matrame)
    AMB_REARM_SEC = 3.0       # ambulance entha sepu kanipinchakapothe malli trigger avvochu

    def __init__(self, routes, controller):
        self.routes = list(routes)
        self.controller = controller
        self.state = "INIT"
        self.current = None
        self.phase_start = 0.0
        self.duration = 0.0
        self.amb_phase = False
        self.last_amb_seen = 0.0
        self.queue = []
        self.reason = ""
        self.warmup_until = None

        self.amb_served = set()       # ee ambulance ki already green ichaam
        self.last_seen = {}           # route -> last time ambulance kanipinchindi
        self.last_amb_route = None
        self.rebuild_queue = False

    # ---------------- helpers ----------------

    def _amb_ok(self, route, confirmed):
        return route in confirmed and route not in self.amb_served

    def _amb_candidates(self, confirmed, exclude=None):
        return [
            r for r in self.routes
            if r != exclude and self._amb_ok(r, confirmed)
        ]

    def _event(self):
        return {
            "type": "GREEN",
            "route": self.current,
            "green_time": int(self.duration),
            "ambulance": self.amb_phase,
            "reason": self.reason,
        }

    def _begin_yellow(self, t):
        self.state = "YELLOW"
        self.phase_start = t

    def _start_green(self, t, avg, confirmed):
        cands = self._amb_candidates(confirmed)

        if cands:
            # ---- ambulance priority ----
            route = max(cands, key=lambda r: avg[r])
            self.amb_phase = True
            self.duration = float(self.controller.AMBULANCE_GREEN_TIME)
            self.reason = "ambulance preemption (first priority)"
            if route in self.queue:
                self.queue.remove(route)
        else:
            # ---- normal cycle ----
            if not self.queue or self.rebuild_queue:
                self.queue = sorted(
                    self.routes, key=lambda r: avg[r], reverse=True
                )
                if self.rebuild_queue and self.last_amb_route in self.queue:
                    # ambulance route ki ippude turn ichaam -> last ki
                    self.queue.remove(self.last_amb_route)
                    self.queue.append(self.last_amb_route)
                    self.reason = "after ambulance: route with most vehicles"
                elif len(self.queue) > 1 and self.queue[0] == self.current:
                    self.queue[0], self.queue[1] = self.queue[1], self.queue[0]
                self.rebuild_queue = False

            route = self.queue.pop(0)
            self.amb_phase = False
            self.duration = float(self.controller.calculate_green_time(avg[route]))
            if not self.reason.startswith("after ambulance"):
                self.reason = "next in the density-ordered cycle"

        self.current = route
        self.state = "GREEN"
        self.phase_start = t
        self.last_amb_seen = t
        event = self._event()
        if not self.amb_phase and self.reason.startswith("after ambulance"):
            self.reason = ""          # next phases ki reset
        return event

    # ---------------- main update ----------------

    def update(self, t, avg, amb_confirmed, amb_seen):
        """Phase marinappudu event dict return chestundi, lekapothe None."""

        # ambulance track: kanipinchina time, malli trigger avvadaniki re-arm
        for r in self.routes:
            if amb_seen.get(r):
                self.last_seen[r] = t
        for r in list(self.amb_served):
            if t - self.last_seen.get(r, -1e9) >= self.AMB_REARM_SEC:
                self.amb_served.discard(r)

        if self.state == "INIT":
            if self.warmup_until is None:
                self.warmup_until = t + self.WARMUP_SEC
            # Ambulance confirm ayithe warm-up aagakunda ventane green
            if t < self.warmup_until and not amb_confirmed:
                return None
            return self._start_green(t, avg, amb_confirmed)

        elapsed = t - self.phase_start

        if self.state == "GREEN":

            if not self.amb_phase:
                # Active green route ki ambulance vaste daanine ambulance phase ga marchu
                if self._amb_ok(self.current, amb_confirmed):
                    self.amb_phase = True
                    self.duration = float(self.controller.AMBULANCE_GREEN_TIME)
                    self.phase_start = t
                    self.last_amb_seen = t
                    self.reason = "ambulance preemption (first priority)"
                    return self._event()

                # Vere route lo ambulance -> yellow, tarvata aa route ki green
                if self._amb_candidates(amb_confirmed, exclude=self.current):
                    self._begin_yellow(t)
                    return None

                if elapsed >= self.duration:
                    self._begin_yellow(t)
                return None

            # ---- ambulance phase ----
            if amb_seen.get(self.current):
                self.last_amb_seen = t

            done = elapsed >= self.duration
            if self.AMB_EARLY_END:
                cleared = (t - self.last_amb_seen) >= self.AMB_CLEAR_SEC
                if elapsed >= self.AMB_MIN_GREEN and cleared:
                    done = True

            if done:
                self.amb_served.add(self.current)
                self.last_amb_route = self.current
                self.rebuild_queue = True
                self.reason = "after ambulance: route with most vehicles"
                self._begin_yellow(t)
            return None

        if self.state == "YELLOW":
            if elapsed >= self.YELLOW_SEC:
                self.state = "ALL_RED"
                self.phase_start = t
            return None

        if self.state == "ALL_RED":
            if elapsed >= self.ALL_RED_SEC:
                return self._start_green(t, avg, amb_confirmed)

        return None

    # ---------------- display helpers ----------------

    def remaining(self, t):
        if self.state == "GREEN":
            return max(0.0, self.duration - (t - self.phase_start))
        if self.state == "YELLOW":
            return max(0.0, self.YELLOW_SEC - (t - self.phase_start))
        return 0.0

    def upcoming(self, avg):
        """Wait chese routes, next turn order lo."""
        order = [r for r in self.queue if r != self.current]
        rest = [
            r for r in sorted(self.routes, key=lambda r: avg[r], reverse=True)
            if r != self.current and r not in order
        ]
        return order + rest

    def snapshot(self, avg, amb_routes):
        out = {}
        for r in self.routes:
            if r == self.current and self.state == "GREEN":
                sig = "GREEN"
            elif r == self.current and self.state == "YELLOW":
                sig = "YELLOW"
            else:
                sig = "RED"

            if r == self.current and self.state == "GREEN":
                gt = int(self.duration)
            else:
                gt = self.controller.calculate_green_time(avg[r])

            out[r] = {
                "vehicles": avg[r],
                "density": self.controller.calculate_density(avg[r]),
                "green_time": gt,
                "signal": sig,
                "ambulance": r in amb_routes,
            }
        return out