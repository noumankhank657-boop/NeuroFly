extends Node
## Non-blocking TCP client to the Python brain (127.0.0.1, newline-delimited JSON).
## It only polls in _process, so the Godot UI never freezes.
signal state_received(state: Dictionary)
signal connection_changed(connected: bool)

var port := 8765
var connected := false
var last_state_msec := 0
var _peer := StreamPeerTCP.new()
var _buffer := ""
var _retry_timer := 0.0
var _connecting := false

func send_command(cmd: Dictionary) -> void:
    if not connected:
        return
    _peer.put_data((JSON.stringify(cmd) + "\n").to_utf8_buffer())

func _process(delta: float) -> void:
    _peer.poll()
    var status := _peer.get_status()
    if status == StreamPeerTCP.STATUS_CONNECTED:
        if not connected:
            connected = true
            connection_changed.emit(true)
        _read_lines()
    else:
        if connected:
            connected = false
            _peer = StreamPeerTCP.new()
            _buffer = ""
            connection_changed.emit(false)
        if status == StreamPeerTCP.STATUS_NONE or status == StreamPeerTCP.STATUS_ERROR:
            _retry_timer -= delta
            if _retry_timer <= 0.0:
                _retry_timer = 1.0
                _peer = StreamPeerTCP.new()
                _peer.connect_to_host("127.0.0.1", port)

func _read_lines() -> void:
    var n := _peer.get_available_bytes()
    if n <= 0:
        return
    _buffer += _peer.get_utf8_string(n)
    var idx := _buffer.find("\n")
    var newest := ""
    while idx >= 0:
        newest = _buffer.substr(0, idx)
        _buffer = _buffer.substr(idx + 1)
        idx = _buffer.find("\n")
    # Only the newest snapshot matters for display (older ones are stale).
    if newest != "":
        var parsed = JSON.parse_string(newest)
        if typeof(parsed) == TYPE_DICTIONARY and parsed.get("schema", "") == "nuerofly.state.v2":
            last_state_msec = Time.get_ticks_msec()
            state_received.emit(parsed)
