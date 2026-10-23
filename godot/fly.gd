class_name NueroFly
extends Node3D

var body_root: Node3D

func _ready() -> void:
    _build_fly()

func _build_fly() -> void:
    # WHAT: assemble a low-poly fruit-fly silhouette from simple primitives.
    # WHY: Phase 8 needs a recognizable scientific embodiment, not a generic cube.
    # OUTPUT: head, eyes, antennae, thorax, abdomen, six legs, and wings.
    # ASSUMPTION: anatomical geometry is intentionally approximate.
    body_root = Node3D.new()
    add_child(body_root)

    _part("Thorax", Vector3(0.0, 0.42, 0.0), Vector3(0.42, 0.34, 0.55), Color(0.28,0.20,0.12))
    _part("Abdomen", Vector3(0.0, 0.38, 0.52), Vector3(0.34, 0.30, 0.68), Color(0.12,0.09,0.07))
    _part("Head", Vector3(0.0, 0.46, -0.48), Vector3(0.34,0.30,0.32), Color(0.20,0.15,0.10))
    _part("LeftEye", Vector3(-0.19,0.49,-0.57), Vector3(0.16,0.16,0.12), Color(0.45,0.05,0.04))
    _part("RightEye", Vector3(0.19,0.49,-0.57), Vector3(0.16,0.16,0.12), Color(0.45,0.05,0.04))

    for side in [-1.0, 1.0]:
        for z in [-0.18, 0.05, 0.27]:
            _leg(side, z)
        _antenna(side)
        _wing(side)

func _part(label: String, pos: Vector3, scale: Vector3, color: Color) -> void:
    var mesh_instance := MeshInstance3D.new()
    mesh_instance.name = label
    var mesh := SphereMesh.new()
    mesh.height = 2.0
    mesh.radius = 1.0
    mesh_instance.mesh = mesh
    mesh_instance.scale = scale
    mesh_instance.position = pos
    var material := StandardMaterial3D.new()
    material.albedo_color = color
    mesh_instance.material_override = material
    body_root.add_child(mesh_instance)

func _leg(side: float, z: float) -> void:
    var leg := MeshInstance3D.new()
    leg.mesh = BoxMesh.new()
    leg.mesh.size = Vector3(0.55,0.06,0.08)
    leg.position = Vector3(0.31 * side, 0.24, z)
    leg.rotation.y = 0.25 * side
    var material := StandardMaterial3D.new()
    material.albedo_color = Color(0.16,0.12,0.09)
    leg.material_override = material
    body_root.add_child(leg)

func _antenna(side: float) -> void:
    var antenna := MeshInstance3D.new()
    antenna.mesh = CylinderMesh.new()
    antenna.mesh.height = 0.34
    antenna.mesh.top_radius = 0.025
    antenna.mesh.bottom_radius = 0.025
    antenna.position = Vector3(0.10 * side, 0.68, -0.63)
    antenna.rotation = Vector3(0.35, 0.0, -0.35 * side)
    body_root.add_child(antenna)

func _wing(side: float) -> void:
    var wing := MeshInstance3D.new()
    wing.mesh = BoxMesh.new()
    wing.mesh.size = Vector3(0.65,0.025,0.24)
    wing.position = Vector3(0.28 * side, 0.63, 0.02)
    wing.rotation.y = 0.12 * side
    var material := StandardMaterial3D.new()
    material.albedo_color = Color(0.55,0.68,0.72,0.45)
    material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
    wing.material_override = material
    body_root.add_child(wing)
