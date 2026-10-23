extends Node3D
## NueroFly main scene. Builds the arena, fly, camera, lights and UI IN CODE,
## so no manual editor work is needed. Godot never decides where the fly goes:
## it only draws state from the Python simulation backend. A* is a conventional
## non-neural planner; LIF and DNa02 remain separate research experiments.
## Coordinates: Godot X/Y/Z == Python X/Y/Z. Only rotation differs:
##   Python yaw>0 turns RIGHT, Godot rotation.y>0 turns LEFT  =>  rotation.y = -yaw.

const FLY_SCRIPT := preload("res://fly.gd")
const CLIENT_SCRIPT := preload("res://backend_client.gd")
const ARENA := 12.0

var client
var fly: Node3D
var fly_target_pos := Vector3(0, 0, 7)
var fly_target_yaw := 0.0
var dest_marker: Node3D
var start_marker: MeshInstance3D
var obstacle_root: Node3D
var traj_mesh: ImmediateMesh
var traj_node: MeshInstance3D
var traj_points: PackedVector3Array = PackedVector3Array()
var route_mesh: ImmediateMesh
var route_node: MeshInstance3D
var camera: Camera3D
var cam_yaw := 0.0
var cam_pitch := 0.9
var cam_dist := 26.0
var dragging := false

var obstacles: Array = [[0.0, 0.0, 1.5], [-1.7, -4.2, 1.0], [1.8, 3.5, 0.9]]
var fields := {}                   # name -> LineEdit
var speed_opt: OptionButton
var status_label: Label
var telemetry_label: Label
var error_label: Label
var conn_label: Label
var last_state := {}
var selftest := false
var shot_mode := false
var shot_done := false
var started_once := false

func _ready() -> void:
    _build_environment()
    _build_markers()
    fly = FLY_SCRIPT.new()
    fly.name = "NueroFly"
    fly.scale = Vector3(0.85, 0.85, 0.85)
    add_child(fly)
    _build_camera()
    _build_ui()
    client = CLIENT_SCRIPT.new()
    for a in OS.get_cmdline_user_args():
        if a.begins_with("--port="):
            client.port = int(a.substr(7))
        if a == "--shot":
            shot_mode = true
            selftest = false
        if a == "--selftest":
            selftest = true
    add_child(client)
    client.state_received.connect(_on_state)
    client.connection_changed.connect(_on_connection)
    _set_status("Backend disconnected - start it with run_nuerofly.ps1", true)
    _draw_obstacles()

# ---------------------------------------------------------------- world
func _mat(color: Color, unshaded := false) -> StandardMaterial3D:
    var m := StandardMaterial3D.new()
    m.albedo_color = color
    if unshaded:
        m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    return m

func _box(parent: Node3D, pos: Vector3, size: Vector3, color: Color) -> void:
    var body := StaticBody3D.new()
    body.position = pos
    parent.add_child(body)
    var mi := MeshInstance3D.new()
    var bm := BoxMesh.new()
    bm.size = size
    mi.mesh = bm
    mi.material_override = _mat(color)
    body.add_child(mi)
    var cs := CollisionShape3D.new()
    var bs := BoxShape3D.new()
    bs.size = size
    cs.shape = bs
    body.add_child(cs)

func _build_environment() -> void:
    var env := Environment.new()
    env.background_mode = Environment.BG_COLOR
    env.background_color = Color(0.09, 0.10, 0.12)
    env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    env.ambient_light_color = Color(0.75, 0.78, 0.85)
    env.ambient_light_energy = 0.7
    var we := WorldEnvironment.new()
    we.environment = env
    add_child(we)
    var sun := DirectionalLight3D.new()
    sun.rotation_degrees = Vector3(-55, 30, 0)
    sun.light_energy = 1.1
    sun.shadow_enabled = true
    add_child(sun)
    _box(self, Vector3(0, -0.1, 0), Vector3(ARENA * 2, 0.2, ARENA * 2), Color(0.20, 0.32, 0.19))
    var wc := Color(0.30, 0.43, 0.29)
    _box(self, Vector3(0, 0.5, -ARENA), Vector3(ARENA * 2, 1.0, 0.2), wc)
    _box(self, Vector3(0, 0.5, ARENA), Vector3(ARENA * 2, 1.0, 0.2), wc)
    _box(self, Vector3(-ARENA, 0.5, 0), Vector3(0.2, 1.0, ARENA * 2), wc)
    _box(self, Vector3(ARENA, 0.5, 0), Vector3(0.2, 1.0, ARENA * 2), wc)
    # Decorative jungle plants stay along the perimeter and have no collision.
    for tree in [Vector2(-10.5, -9.0), Vector2(10.0, -8.0), Vector2(-10.5, 1.0), Vector2(10.5, 3.0), Vector2(-9.0, 10.0), Vector2(9.5, 10.5)]:
        _decorative_tree(tree.x, tree.y)
    # ground grid every 2 units (spatial reference)
    var grid := ImmediateMesh.new()
    grid.surface_begin(Mesh.PRIMITIVE_LINES, _mat(Color(0.45, 0.5, 0.5), true))
    for i in range(-6, 7):
        var v := float(i) * 2.0
        grid.surface_add_vertex(Vector3(v, 0.01, -ARENA))
        grid.surface_add_vertex(Vector3(v, 0.01, ARENA))
        grid.surface_add_vertex(Vector3(-ARENA, 0.01, v))
        grid.surface_add_vertex(Vector3(ARENA, 0.01, v))
    grid.surface_end()
    var gi := MeshInstance3D.new()
    gi.mesh = grid
    add_child(gi)
    obstacle_root = Node3D.new()
    add_child(obstacle_root)
    traj_mesh = ImmediateMesh.new()
    traj_node = MeshInstance3D.new()
    traj_node.mesh = traj_mesh
    add_child(traj_node)
    route_mesh = ImmediateMesh.new()
    route_node = MeshInstance3D.new()
    route_node.mesh = route_mesh
    add_child(route_node)

func _decorative_tree(x: float, z: float) -> void:
    var tree := Node3D.new()
    tree.position = Vector3(x, 0.0, z)
    add_child(tree)
    var trunk := MeshInstance3D.new()
    var trunk_mesh := CylinderMesh.new()
    trunk_mesh.height = 2.1
    trunk_mesh.top_radius = 0.22
    trunk_mesh.bottom_radius = 0.32
    trunk.mesh = trunk_mesh
    trunk.position.y = 1.05
    trunk.material_override = _mat(Color(0.30, 0.19, 0.10))
    tree.add_child(trunk)
    var crown := MeshInstance3D.new()
    var crown_mesh := SphereMesh.new()
    crown_mesh.radius = 1.05
    crown_mesh.height = 2.1
    crown.mesh = crown_mesh
    crown.position.y = 2.25
    crown.material_override = _mat(Color(0.16, 0.38, 0.18))
    tree.add_child(crown)

func _build_markers() -> void:
    dest_marker = Node3D.new()
    add_child(dest_marker)
    var disc := MeshInstance3D.new()
    var cm := CylinderMesh.new()
    cm.height = 0.06
    cm.top_radius = 0.5
    cm.bottom_radius = 0.5
    disc.mesh = cm
    disc.material_override = _mat(Color(1.0, 0.75, 0.1), true)
    dest_marker.add_child(disc)
    var pole := MeshInstance3D.new()
    var pm := CylinderMesh.new()
    pm.height = 2.2
    pm.top_radius = 0.04
    pm.bottom_radius = 0.04
    pole.mesh = pm
    pole.position.y = 1.1
    pole.material_override = _mat(Color(1.0, 0.75, 0.1), true)
    dest_marker.add_child(pole)
    start_marker = MeshInstance3D.new()
    var sm := CylinderMesh.new()
    sm.height = 0.04
    sm.top_radius = 0.4
    sm.bottom_radius = 0.4
    start_marker.mesh = sm
    start_marker.material_override = _mat(Color(0.25, 0.6, 1.0), true)
    add_child(start_marker)
    dest_marker.position = Vector3(0, 0.03, -7)
    start_marker.position = Vector3(0, 0.02, 7)

func _draw_obstacles() -> void:
    for c in obstacle_root.get_children():
        c.queue_free()
    for o in obstacles:
        var clearance := MeshInstance3D.new()
        var clearance_mesh := CylinderMesh.new()
        clearance_mesh.height = 0.025
        clearance_mesh.top_radius = float(o[2]) + 0.45
        clearance_mesh.bottom_radius = float(o[2]) + 0.45
        clearance.mesh = clearance_mesh
        clearance.position = Vector3(float(o[0]), 0.025, float(o[1]))
        var clearance_material := _mat(Color(0.92, 0.40, 0.22, 0.25))
        clearance_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
        clearance.material_override = clearance_material
        obstacle_root.add_child(clearance)
        var mi := MeshInstance3D.new()
        var cm := CylinderMesh.new()
        cm.height = 1.2
        cm.top_radius = float(o[2])
        cm.bottom_radius = float(o[2])
        mi.mesh = cm
        mi.position = Vector3(float(o[0]), 0.6, float(o[1]))
        mi.material_override = _mat(Color(0.75, 0.25, 0.25))
        obstacle_root.add_child(mi)
        # Canopy sits above the ground-plane collision model; the trunk radius
        # is the obstacle radius sent to Python.
        var crown := MeshInstance3D.new()
        var sm := SphereMesh.new()
        sm.radius = float(o[2]) * 0.78
        sm.height = float(o[2]) * 1.56
        crown.mesh = sm
        crown.position = Vector3(float(o[0]), 1.15, float(o[1]))
        crown.material_override = _mat(Color(0.22, 0.48, 0.19))
        obstacle_root.add_child(crown)

func _build_camera() -> void:
    camera = Camera3D.new()
    camera.name = "ScientificCamera"
    camera.far = 200.0
    add_child(camera)
    camera.current = true
    _reset_camera()

func _reset_camera() -> void:
    cam_yaw = 0.0
    cam_pitch = 0.9
    cam_dist = 26.0
    _update_camera()

func _update_camera() -> void:
    var offset := Vector3(sin(cam_yaw) * cos(cam_pitch), sin(cam_pitch), cos(cam_yaw) * cos(cam_pitch)) * cam_dist
    camera.position = offset
    camera.look_at(Vector3.ZERO, Vector3.UP)

func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventMouseButton:
        if event.button_index == MOUSE_BUTTON_RIGHT:
            dragging = event.pressed
        elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
            cam_dist = max(6.0, cam_dist - 1.5)
            _update_camera()
        elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
            cam_dist = min(60.0, cam_dist + 1.5)
            _update_camera()
    elif event is InputEventMouseMotion and dragging:
        cam_yaw -= event.relative.x * 0.01
        cam_pitch = clamp(cam_pitch + event.relative.y * 0.01, 0.15, 1.5)
        _update_camera()
    elif event is InputEventKey and event.pressed and event.keycode == KEY_R:
        _reset_camera()

# ---------------------------------------------------------------- UI
func _build_ui() -> void:
    var layer := CanvasLayer.new()
    add_child(layer)
    var panel := PanelContainer.new()
    panel.position = Vector2(10, 10)
    layer.add_child(panel)
    var scroll := ScrollContainer.new()
    scroll.custom_minimum_size = Vector2(350, 680)
    scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    panel.add_child(scroll)
    var box := VBoxContainer.new()
    box.custom_minimum_size = Vector2(330, 0)
    scroll.add_child(box)
    var title := Label.new()
    title.text = "Digital Fly: 3D navigation simulation"
    box.add_child(title)
    conn_label = Label.new()
    box.add_child(conn_label)
    status_label = Label.new()
    box.add_child(status_label)
    error_label = Label.new()
    error_label.add_theme_color_override("font_color", Color(1, 0.45, 0.4))
    box.add_child(error_label)
    box.add_child(_section("Start position (x, z) and heading"))
    box.add_child(_row([["start_x", "0"], ["start_z", "7"], ["yaw_deg", "0"]]))
    box.add_child(_section("Destination (x, z)   [y fixed at ground]"))
    box.add_child(_row([["dest_x", "0"], ["dest_z", "-7"]]))
    box.add_child(_section("Goal tolerance (arrival radius)"))
    box.add_child(_row([["goal_tolerance", "0.5"]]))
    box.add_child(_section("Experimental reinforcement learning / seed"))
    var mrow := HBoxContainer.new()
    var controller_name := Label.new()
    controller_name.text = "Tabular Q-learning (non-biological)"
    mrow.add_child(controller_name)
    var seed_edit := LineEdit.new()
    seed_edit.text = "1"
    seed_edit.placeholder_text = "seed"
    seed_edit.custom_minimum_size = Vector2(60, 0)
    fields["seed"] = seed_edit
    mrow.add_child(seed_edit)
    speed_opt = OptionButton.new()
    for s in ["1x", "2x", "4x", "8x"]:
        speed_opt.add_item(s)
    mrow.add_child(speed_opt)
    box.add_child(mrow)
    var b1 := HBoxContainer.new()
    b1.add_child(_btn("Start", _on_start))
    b1.add_child(_btn("Pause", func(): _send({"cmd": "pause"})))
    b1.add_child(_btn("Resume", func(): _send({"cmd": "resume"})))
    b1.add_child(_btn("Reset", _on_reset))
    box.add_child(b1)
    box.add_child(_btn("Clear learned memory + reset", _on_clear_memory))
    box.add_child(_section("Obstacle (x, z, radius)"))
    box.add_child(_row([["obs_x", "0"], ["obs_z", "0"], ["obs_r", "1.5"]]))
    var b2 := HBoxContainer.new()
    b2.add_child(_btn("Add obstacle", _on_add_obstacle))
    b2.add_child(_btn("Remove last", _on_remove_obstacle))
    b2.add_child(_btn("Clear", func(): obstacles.clear(); _draw_obstacles(); _on_reset()))
    box.add_child(b2)
    var b3 := HBoxContainer.new()
    b3.add_child(_btn("Reset camera (R)", _reset_camera))
    var cb := CheckBox.new()
    cb.text = "Trajectory"
    cb.button_pressed = true
    cb.toggled.connect(func(on: bool): traj_node.visible = on)
    b3.add_child(cb)
    box.add_child(b3)
    box.add_child(_section("Telemetry (from Python backend)"))
    telemetry_label = Label.new()
    box.add_child(telemetry_label)

func _section(t: String) -> Label:
    var l := Label.new()
    l.text = t
    l.add_theme_color_override("font_color", Color(0.65, 0.8, 1.0))
    return l

func _row(items: Array) -> HBoxContainer:
    var row := HBoxContainer.new()
    for it in items:
        var le := LineEdit.new()
        le.text = it[1]
        le.custom_minimum_size = Vector2(70, 0)
        le.tooltip_text = it[0]
        fields[it[0]] = le
        row.add_child(le)
    return row

func _btn(t: String, cb: Callable) -> Button:
    var b := Button.new()
    b.text = t
    b.pressed.connect(cb)
    return b

func _send(cmd: Dictionary) -> void:
    # Never fail silently: tell the user when the Python backend is not running.
    if not client.connected:
        _set_status("NOT CONNECTED. Start the Python backend first (see README), then press the button again.", true)
        return
    client.send_command(cmd)

func _num(name: String) -> float:
    var s: String = fields[name].text.strip_edges()
    if not s.is_valid_float():
        _set_status("Invalid number in field '%s'" % name, true)
        return NAN
    return s.to_float()

func _config_message(cmd: String) -> Dictionary:
    var vals := {}
    for k in ["start_x", "start_z", "yaw_deg", "dest_x", "dest_z", "goal_tolerance"]:
        var v := _num(k)
        if is_nan(v):
            return {}
        vals[k] = v
    return {"cmd": cmd, "start": [vals.start_x, vals.start_z], "yaw_deg": vals.yaw_deg,
        "destination": [vals.dest_x, vals.dest_z],
        "tolerance": vals.goal_tolerance,
        "mode": "learning", "seed": int(fields["seed"].text) if fields["seed"].text.is_valid_int() else 1,
        "obstacles": obstacles, "speed": [1, 2, 4, 8][speed_opt.selected]}

func _on_start() -> void:
    var m := _config_message("configure")
    if m.is_empty():
        return
    traj_points.clear()
    _send(m)
    _send({"cmd": "start"})

func _on_reset() -> void:
    var m := _config_message("configure")
    if m.is_empty():
        return
    traj_points.clear()
    _redraw_traj()
    _send(m)
    _send({"cmd": "reset"})

func _on_clear_memory() -> void:
    traj_points.clear()
    _redraw_traj()
    _send({"cmd": "clear_memory"})

func _on_add_obstacle() -> void:
    var x := _num("obs_x"); var z := _num("obs_z"); var r := _num("obs_r")
    if is_nan(x) or is_nan(z) or is_nan(r) or r <= 0.0:
        _set_status("Obstacle needs numbers and radius > 0", true)
        return
    obstacles.append([x, z, r])
    _draw_obstacles()
    _on_reset()

func _on_remove_obstacle() -> void:
    if obstacles.size() > 0:
        obstacles.pop_back()
        _draw_obstacles()
        _on_reset()

func _set_status(t: String, is_error := false) -> void:
    status_label.text = t
    status_label.add_theme_color_override("font_color", Color(1, 0.5, 0.4) if is_error else Color(0.6, 1, 0.7))

# ---------------------------------------------------------------- backend events
func _on_connection(ok: bool) -> void:
    if ok:
        _set_status("Backend connected")
        _on_reset()   # push current UI values so the backend shows a valid default arena
    else:
        _set_status("Backend disconnected - start it with run_nuerofly.ps1", true)

func _on_state(s: Dictionary) -> void:
    last_state = s
    var p: Array = s["position"]
    fly_target_pos = Vector3(p[0], p[1], p[2])
    fly_target_yaw = -float(s["yaw_rad"])     # sign flip, see header
    var d: Array = s["destination"]
    dest_marker.position = Vector3(d[0], 0.03, d[2])
    var st: Array = s["start"]
    start_marker.position = Vector3(st[0], 0.02, st[2])
    error_label.text = "" if s.get("error") == null else str(s["error"])
    _set_status(str(s["message"]), s["status"] in ["collision", "avoiding", "timeout", "error", "no_route", "route_blocked", "stuck", "failed"])
    _draw_route(s.get("route", []))
    if s["status"] in ["running", "exploring", "navigating", "learning", "avoiding"]:
        if traj_points.is_empty() or traj_points[traj_points.size() - 1].distance_to(fly_target_pos) > 0.05:
            traj_points.append(Vector3(p[0], 0.05, p[2]))
            if traj_points.size() > 4000:
                traj_points.remove_at(0)
            _redraw_traj()
    telemetry_label.text = "Controller: %s\nStatus: %s   step %d   t=%.2fs\nPos x=%.2f y=%.2f z=%.2f\nHeading %.1f deg  err %.1f deg\nDest  x=%.1f z=%.1f   dist %.2f\nSteer L=%.2f R=%.2f F=%.2f\nSpeed %.2f   yaw rate %.2f\nBlocked moves: %d   Path %.2f" % [
        s["controller_note"], s["status"], int(s["step"]), float(s["time"]),
        p[0], p[1], p[2], rad_to_deg(float(s["yaw_rad"])), rad_to_deg(float(s["heading_error"])),
        d[0], d[2], float(s["distance"]),
        s["steering"]["left"], s["steering"]["right"], s["steering"]["forward"],
        float(s["speed"]), float(s["yaw_rate"]),
        int(s.get("collision_count", 0)), float(s["path_length"])]
    telemetry_label.text += "\nGoal tolerance %.2f   fly radius %.2f + safety %.2f" % [float(s.get("goal_tolerance", 0.5)), float(s.get("fly_radius", 0.3)), float(s.get("safety_margin", 0.15))]
    var learning: Dictionary = s.get("learning", {})
    telemetry_label.text += "\nLearning episodes %d   states %d   seed %d   action %s" % [int(learning.get("episodes", 0)), int(learning.get("states", 0)), int(learning.get("seed", 1)), str(learning.get("action", "-"))]
    if s.get("mode") == "planned":
        telemetry_label.text += "\nA* waypoints: %d   next: %d   replans: %d" % [s.get("route", []).size(), int(s.get("route_index", 0)), int(s.get("route_replans", 0))]
    if selftest and s["status"] in ["reached", "collision", "timeout", "stuck", "failed"]:
        print("SELFTEST status=%s dist=%.2f steps=%d" % [s["status"], float(s["distance"]), int(s["step"])])
        get_tree().quit()

func _redraw_traj() -> void:
    traj_mesh.clear_surfaces()
    if traj_points.size() < 2:
        return
    traj_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP, _mat(Color(0.3, 1.0, 0.6), true))
    for pt in traj_points:
        traj_mesh.surface_add_vertex(pt)
    traj_mesh.surface_end()

func _draw_route(points: Array) -> void:
    route_mesh.clear_surfaces()
    if points.size() < 2:
        return
    route_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP, _mat(Color(0.25, 0.75, 1.0), true))
    for pt in points:
        route_mesh.surface_add_vertex(Vector3(float(pt[0]), 0.08, float(pt[1])))
    route_mesh.surface_end()

func _process(delta: float) -> void:
    if client.connected:
        conn_label.text = "BACKEND: CONNECTED"
        conn_label.add_theme_color_override("font_color", Color(0.4, 1, 0.5))
    else:
        conn_label.text = "BACKEND: NOT CONNECTED (nothing will move)"
        conn_label.add_theme_color_override("font_color", Color(1, 0.4, 0.3))
    # Fly moves ONLY toward backend-published state. If the backend stops
    # talking, we flag it instead of letting the fly drift.
    if client.connected and client.last_state_msec > 0 and Time.get_ticks_msec() - client.last_state_msec > 2000:
        _set_status("Backend not responding (no data for 2 s)", true)
    var k: float = min(1.0, delta * 20.0)
    fly.position = fly.position.lerp(fly_target_pos, k)
    fly.rotation.y = lerp_angle(fly.rotation.y, fly_target_yaw, k)
    if shot_mode and Time.get_ticks_msec() > 8500 and not shot_done:
        shot_done = true
        _save_shot()
        get_tree().quit()
    if selftest and Time.get_ticks_msec() > 80000:
        print("SELFTEST timeout waiting for trial completion")
        get_tree().quit()
    if (selftest or shot_mode) and client.connected and last_state.is_empty() == false and last_state["status"] == "ready" and not started_once and Time.get_ticks_msec() > 2000:
        started_once = true
        _on_start()

func _save_shot() -> void:
    # Developer aid used by automated checks: save what the viewport shows.
    var img := get_viewport().get_texture().get_image()
    img.save_png("user://shot.png")
    print("SHOT ", ProjectSettings.globalize_path("user://shot.png"))
