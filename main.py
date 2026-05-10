import threading
import time

import speech_recognition as sr
import sounddevice  # noqa: F401
from flask import Flask, redirect, render_template, request, url_for
from pyniryo import (
    ConveyorDirection,
    NiryoRobot,
    ObjectColor,
    ObjectShape,
    PinID,
    PinState,
    PoseObject,
)

ROBOT_IP        = "169.254.200.200"
PICK_WORKSPACE  = "con"
SENSOR_PIN      = PinID.DI5
MAX_CATCH_COUNT = 9
CONVEYOR_SPEED  = 70
HEIGHT_OFFSET   = 0.01

POSES: dict[str, PoseObject] = {
    "alpha": PoseObject(-0.2233, -0.3543, 0.0060,  2.2191, 1.4832,  0.5268),
    "beta":  PoseObject(-0.2053,  0.2164, 0.0294,  1.5675, 1.3544, -1.8412),
    "scrap": PoseObject(-0.0062, -0.2469, 0.1054,  2.1222, 1.2687,  1.2565),
    "pick":  PoseObject( 0.1535,  0.1172, 0.3049, -2.3984, 1.2315, -2.0885),
}

COLOR_MAP: dict[str, ObjectColor] = {
    "green": ObjectColor.GREEN,
    "blue":  ObjectColor.BLUE,
    "red":   ObjectColor.RED,
}
SHAPE_MAP: dict[str, ObjectShape] = {
    "circle": ObjectShape.CIRCLE,
    "square": ObjectShape.SQUARE,
}
VALID_PLACES   = frozenset({"alpha", "beta", "scrap"})
VOICE_COMMANDS = {"setup", "start", "stop"}

DEST_TO_POSE: dict[str, str] = {
    "Alpha Zone":       "alpha",
    "Beta Zone":        "beta",
    "Vision Workspace": "scrap",
}

state_lock = threading.Lock()

shared_state: dict = {
    "conveyor_running": False,
    "catch_count": 0,
    "target_color": None,
    "target_shape": None,
    "target_place": None,
    "sorting_rules": {
        "green_circle": "Alpha Zone",
        "green_square": "Beta Zone",
        "blue_circle":  "Vision Workspace",
        "blue_square":  "Beta Zone",
        "red_circle":   "Alpha Zone",
        "red_square":   "Vision Workspace",
    },
    "log": [],
}


def log(msg: str) -> None:
    print(msg)
    with state_lock:
        shared_state["log"].append(msg)
        if len(shared_state["log"]) > 50:
            shared_state["log"] = shared_state["log"][-50:]


def _color_to_str(color: ObjectColor) -> str:
    return {ObjectColor.GREEN: "green", ObjectColor.BLUE: "blue", ObjectColor.RED: "red"}.get(color, "unknown")


def _shape_to_str(shape: ObjectShape) -> str:
    return {ObjectShape.CIRCLE: "circle", ObjectShape.SQUARE: "square"}.get(shape, "unknown")


class RobotSorter:

    def __init__(self, robot: NiryoRobot, conveyor_id) -> None:
        self.robot            = robot
        self.conveyor_id      = conveyor_id
        self._conveyor_hw_on  = False

    def _hw_start_conveyor(self) -> None:
        self.robot.run_conveyor(self.conveyor_id, speed=CONVEYOR_SPEED, direction=ConveyorDirection.FORWARD)
        self._conveyor_hw_on = True
        log("conveyor started.")

    def _hw_stop_conveyor(self) -> None:
        self.robot.stop_conveyor(self.conveyor_id)
        self._conveyor_hw_on = False
        log("conveyor stopped.")

    def sensor_triggered(self) -> bool:
        return self.robot.digital_read(SENSOR_PIN) == PinState.LOW

    def _place(self, pose_key: str) -> None:
        self.robot.move_pose(POSES[pose_key])
        self.robot.release_with_tool()
        log(f"object placed at '{pose_key}'.")

    def process_one_object(self) -> None:
        self.robot.move_pose(POSES["pick"])
        obj_found, shape_ret, color_ret = self.robot.vision_pick(
            PICK_WORKSPACE,
            shape=ObjectShape.ANY,
            color=ObjectColor.ANY,
            height_offset=HEIGHT_OFFSET,
        )

        if not obj_found:
            log("  ! No object detected by camera.")
            return

        color_name = _color_to_str(color_ret)
        shape_name = _shape_to_str(shape_ret)
        log(f"  ~ Detected: shape={shape_name}, color={color_name}")

        with state_lock:
            t_color = shared_state["target_color"]
            t_shape = shared_state["target_shape"]
            t_place = shared_state["target_place"]
            rules   = shared_state["sorting_rules"]

        if t_color is not None and t_shape is not None and t_place is not None:
            if shape_ret == t_shape and color_ret == t_color:
                self._place(t_place)
            else:
                log("  Mismatch — sending to scrap.")
                self._place("scrap")
        else:
            object_key = f"{color_name}_{shape_name}"
            dest_name  = rules.get(object_key, "Vision Workspace")
            pose_key   = DEST_TO_POSE.get(dest_name, "scrap")
            log(f"  Web rule: {object_key} → {dest_name} ({pose_key})")
            self._place(pose_key)

        self.robot.wait(2.0)
        with state_lock:
            shared_state["catch_count"] += 1
            count = shared_state["catch_count"]
        log(f"  Caught: {count}/{MAX_CATCH_COUNT}")

    def control_loop(self) -> None:
        while True:
            with state_lock:
                should_run = shared_state["conveyor_running"]
                count      = shared_state["catch_count"]

            if count >= MAX_CATCH_COUNT:
                if self._conveyor_hw_on:
                    self._hw_stop_conveyor()
                with state_lock:
                    shared_state["conveyor_running"] = False
                log(f"  Max catch count ({MAX_CATCH_COUNT}) reached. Stopping.")
                time.sleep(1.0)
                continue

            if should_run:
                if not self._conveyor_hw_on:
                    self._hw_start_conveyor()
                if self.sensor_triggered():
                    self._hw_stop_conveyor()
                    self.process_one_object()
                    with state_lock:
                        still_run = shared_state["conveyor_running"]
                    if still_run:
                        self._hw_start_conveyor()
            else:
                if self._conveyor_hw_on:
                    self._hw_stop_conveyor()

            time.sleep(0.2)

    def shutdown(self) -> None:
        if self._conveyor_hw_on:
            self._hw_stop_conveyor()
        self.robot.go_to_sleep()
        log("  Robot is sleeping. Byeeee!(＾Ｕ＾)ノ")


def listen(
    recognizer: sr.Recognizer,
    microphone: sr.Microphone,
    prompt: str,
    *,
    listen_duration: float = 5.0,
    ambient_duration: float = 0.5,
) -> str:
    print(f"\n  Listening  {prompt}")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)
        print("     (speak now...)")
        audio = recognizer.record(source, duration=listen_duration)
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"     Heard: '{text}'")
        return text
    except sr.UnknownValueError:
        print("     Could not understand speech.")
    except sr.RequestError as exc:
        print(f"     Recognition service error: {exc}")
    return ""


def ask_until_valid(
    recognizer: sr.Recognizer,
    microphone: sr.Microphone,
    prompt: str,
    valid_map: dict,
    retries: int = 3,
) -> object | None:
    for attempt in range(1, retries + 1):
        word   = listen(recognizer, microphone, prompt)
        result = valid_map.get(word)
        if result is not None:
            return result
        remaining = retries - attempt
        if remaining:
            print(f"     Invalid. Attempts left: {remaining}. Valid: {', '.join(valid_map)}")
    print("     No valid input received.")
    return None


def handle_setup(rec: sr.Recognizer, mic: sr.Microphone) -> None:
    print("\n-------------- setup")

    color = ask_until_valid(rec, mic, f"Say COLOR: {', '.join(COLOR_MAP)}", COLOR_MAP)
    if color is None:
        print("  Setup cancelled (no color)."); return

    shape = ask_until_valid(rec, mic, f"Say SHAPE: {', '.join(SHAPE_MAP)}", SHAPE_MAP)
    if shape is None:
        print("  Setup cancelled (no shape)."); return

    place = ask_until_valid(rec, mic, f"Say PLACE: {', '.join(VALID_PLACES)}", {p: p for p in VALID_PLACES})
    if place is None:
        print("  Setup cancelled (no place)."); return

    with state_lock:
        shared_state["target_color"] = color
        shared_state["target_shape"] = shape
        shared_state["target_place"] = place
    log(f"  Voice setup: color={color.name}, shape={shape.name}, place={place}")


def voice_loop(rec: sr.Recognizer, mic: sr.Microphone) -> None:
    while True:
        with state_lock:
            count = shared_state["catch_count"]
        if count >= MAX_CATCH_COUNT:
            time.sleep(1.0)
            continue

        command = listen(rec, mic, prompt=f"Command: {' | '.join(sorted(VOICE_COMMANDS))}")

        if command == "setup":
            handle_setup(rec, mic)
        elif command == "start":
            with state_lock:
                ready = (
                    shared_state["target_color"] is not None
                    and shared_state["target_shape"] is not None
                    and shared_state["target_place"] is not None
                )
            if not ready:
                print("  ! Run 'setup' first.")
                continue
            with state_lock:
                shared_state["conveyor_running"] = True
            log("  Voice: conveyor started.")
        elif command == "stop":
            with state_lock:
                shared_state["conveyor_running"] = False
            log("  Voice: conveyor stopped.")
        elif command:
            print(f"  ??? Unknown command: '{command}'")


app = Flask(__name__)


@app.route("/")
def home():
    with state_lock:
        conveyor_running = shared_state["conveyor_running"]
        sorting_rules    = shared_state["sorting_rules"]
        catch_count      = shared_state["catch_count"]
        log_lines        = list(shared_state["log"])
        t_color          = shared_state["target_color"]
        t_shape          = shared_state["target_shape"]
        t_place          = shared_state["target_place"]

    voice_status = {
        "color": t_color.name if t_color else "—",
        "shape": t_shape.name if t_shape else "—",
        "place": t_place or "—",
    }
    return render_template(
        "index.html",
        conveyor_running=conveyor_running,
        sorting_rules=sorting_rules,
        catch_count=catch_count,
        max_catch=MAX_CATCH_COUNT,
        log_lines=log_lines[-20:],
        voice_status=voice_status,
    )


@app.route("/control", methods=["POST"])
def control():
    action = request.form["action"]
    with state_lock:
        if action == "start":
            shared_state["conveyor_running"] = True
        elif action == "stop":
            shared_state["conveyor_running"] = False
    return redirect(url_for("home"))


@app.route("/set_sorting_rules", methods=["POST"])
def set_sorting_rules():
    keys = ["green_circle", "green_square", "blue_circle", "blue_square", "red_circle", "red_square"]
    with state_lock:
        for k in keys:
            shared_state["sorting_rules"][k] = request.form[k]
    return redirect(url_for("home"))


@app.route("/reset_count", methods=["POST"])
def reset_count():
    with state_lock:
        shared_state["catch_count"] = 0
    return redirect(url_for("home"))


def main() -> None:
    log("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)
    robot.clear_collision_detected()
    robot.calibrate_auto()
    robot.update_tool()
    conveyor_id = robot.set_conveyor()
    log("Robot ready.\n")

    sorter = RobotSorter(robot, conveyor_id)
    rec    = sr.Recognizer()
    mic    = sr.Microphone()

    threading.Thread(target=sorter.control_loop, daemon=True).start()
    threading.Thread(target=voice_loop, args=(rec, mic), daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, use_reloader=False), daemon=True).start()

    log("Web interface running at http://0.0.0.0:5000")
    log("Voice commands: setup | start | stop\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("\nStopped by user.")
    finally:
        sorter.shutdown()


if __name__ == "__main__":
    main()
