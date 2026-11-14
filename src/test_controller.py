from traffic_controller import TrafficSignalController


controller = TrafficSignalController()


def show(title, selected_route, decisions):
    print(f"\n========== {title} ==========\n")
    for route, data in decisions.items():
        amb = " | AMBULANCE" if data["ambulance"] else ""
        print(
            f"{route}: {data['vehicles']} vehicles | {data['density']} | "
            f"{data['green_time']} sec | {data['signal']}{amb}"
        )
    print("\nSelected Route:", selected_route)
    print("=" * 40)


# TEST 1: Normal traffic (14 vehicles = 20 sec)
selected, d = controller.get_signal_decision(
    {"Route 1": 7, "Route 2": 14, "Route 3": 10, "Route 4": 5}
)
show("TEST 1: NORMAL TRAFFIC", selected, d)
assert selected == "Route 2"
assert d["Route 2"]["green_time"] == 20

# TEST 2: 21-25 vehicles = 40 sec
selected, d = controller.get_signal_decision(
    {"Route 1": 7, "Route 2": 23, "Route 3": 10, "Route 4": 5}
)
show("TEST 2: 21-25 VEHICLES", selected, d)
assert d["Route 2"]["green_time"] == 40

# TEST 3: more than 30 vehicles = 50 sec
selected, d = controller.get_signal_decision(
    {"Route 1": 7, "Route 2": 35, "Route 3": 10, "Route 4": 5}
)
show("TEST 3: MORE THAN 30 VEHICLES", selected, d)
assert d["Route 2"]["green_time"] == 50
assert d["Route 2"]["density"] == "VERY HIGH"

# TEST 4: Ambulance low-traffic route lo (priority)
selected, d = controller.get_signal_decision(
    {"Route 1": 7, "Route 2": 35, "Route 3": 3, "Route 4": 5},
    ambulance_lanes=["Route 3"],
)
show("TEST 4: AMBULANCE PRIORITY", selected, d)
assert selected == "Route 3"
assert d["Route 3"]["green_time"] == 40
assert d["Route 2"]["signal"] == "RED"

# TEST 5: Rendu routes lo ambulance
selected, d = controller.get_signal_decision(
    {"Route 1": 12, "Route 2": 4, "Route 3": 3, "Route 4": 5},
    ambulance_lanes=["Route 2", "Route 4"],
)
show("TEST 5: TWO AMBULANCES", selected, d)
assert selected == "Route 4"

print("\nAll tests passed")