class TrafficSignalController:

    AMBULANCE_GREEN_TIME = 40

    # (max vehicles, green time sec)
    GREEN_TIERS = [
        (10, 10),
        (15, 20),
        (20, 30),
        (25, 40),
        (30, 45),
    ]
    MAX_GREEN_TIME = 50       # 30 kanna ekkuva vehicles

    def calculate_density(self, count):
        if count <= 10:
            return "LOW"
        elif count <= 20:
            return "MEDIUM"
        elif count <= 30:
            return "HIGH"
        return "VERY HIGH"

    def calculate_green_time(self, count):
        for max_count, seconds in self.GREEN_TIERS:
            if count <= max_count:
                return seconds
        return self.MAX_GREEN_TIME

    def select_lane(self, lane_counts, ambulance_lanes=None):
        if ambulance_lanes:
            return max(ambulance_lanes, key=lambda l: lane_counts.get(l, 0))
        return max(lane_counts, key=lane_counts.get)

    def get_signal_decision(self, lane_counts, ambulance_lanes=None):
        ambulance_lanes = [l for l in (ambulance_lanes or []) if l in lane_counts]
        selected_lane = self.select_lane(lane_counts, ambulance_lanes)

        decisions = {}
        for lane, count in lane_counts.items():
            is_selected = (lane == selected_lane)
            has_ambulance = lane in ambulance_lanes

            if is_selected and has_ambulance:
                green_time = self.AMBULANCE_GREEN_TIME
            else:
                green_time = self.calculate_green_time(count)

            decisions[lane] = {
                "vehicles": count,
                "density": self.calculate_density(count),
                "green_time": green_time,
                "signal": "GREEN" if is_selected else "RED",
                "ambulance": has_ambulance
            }

        return selected_lane, decisions