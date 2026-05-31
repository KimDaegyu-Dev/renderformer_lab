# NOTE: install bpy (4.0.0) and bpy_helper (0.0.8) before usage
# pip install bpy==4.0.0 bpy_helper==0.0.8 --extra-index-url https://download.blender.org/pypi/

"""
Blender/Cycles scene generation + Stage-0 GT export.

Added for Stage 0 probing:
  - Cycles render passes: Combined, Diffuse Direct, Diffuse Indirect, Object Index
  - Optional triangle-object split so Object Index becomes triangle_id_buffer
  - Per-view H5 export with:
      I_total, I_direct, I_indirect, triangle_id_buffer,
      L_direct_tri, L_indirect_tri, L_total_tri,
      tri_visible_count, tri_valid_mask, c2w

Typical Stage-0 command:

python to_blend_stage0.py scene.json \
  --output_dir out \
  --mesh_path out/scene.obj \
  --save_img \
  --stage0_gt \
  --stage0_split_triangles \
  --resolution 256 \
  --spp 1024

Caveat:
  Object Index pass is object-level. For true triangle_id_buffer, use
  --stage0_split_triangles so each triangle is converted into a separate
  Blender object with pass_index = triangle_id + 1.
"""
"""
Stage 0 GT를 만들려면 이렇게 실행하면 돼.

python to_blend_stage0.py scene.json \
  --output_dir out \
  --mesh_path out/scene.obj \
  --save_img \
  --stage0_gt \
  --stage0_split_triangles \
  --resolution 256 \
  --spp 1024

핵심 옵션은 이거야.

--stage0_gt

direct / indirect / object index pass를 뽑고 H5로 저장하는 옵션.

--stage0_split_triangles

각 triangle을 Blender object 하나로 쪼개서 Object Index pass가 곧 triangle_id_buffer가 되게 하는 옵션.

이 옵션을 안 쓰면 Object Index는 triangle ID가 아니라 object ID가 돼. Stage 0에서는 보통 반드시 켜는 게 맞아.
"""
import bpy

import os
import json
import glob
import tempfile
import sys
import numpy as np
from contextlib import contextmanager, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import imageio
except ImportError:
    imageio = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

try:
    from dacite import from_dict, Config
except ImportError:
    from_dict = None
    Config = None

try:
    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.material import create_specular_roughness_material, create_white_emmissive_material
    from bpy_helper.scene import reset_scene, import_3d_model, scene_meshes
    from bpy_helper.utils import stdout_redirected
    from bpy_helper.io import save_blend_file
    USING_BPY_HELPER = True
except ImportError:
    USING_BPY_HELPER = False
    @contextmanager
    def stdout_redirected():
        with open(os.devnull, "w") as f, redirect_stdout(f):
            yield

    def reset_scene() -> None:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def scene_meshes():
        return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    def import_3d_model(path: str) -> None:
        before = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=os.path.abspath(path))
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        for obj in imported:
            obj.name = os.path.splitext(os.path.basename(path))[0]

    def save_blend_file(path: str) -> None:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))

    def look_at_to_c2w(camera_position, target_position=[0.0, 0.0, 0.0], up_dir=[0.0, 0.0, 1.0]) -> np.ndarray:
        camera_direction = np.array(camera_position) - np.array(target_position)
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        camera_right = np.cross(np.array(up_dir), camera_direction)
        camera_right = camera_right / np.linalg.norm(camera_right)
        camera_up = np.cross(camera_direction, camera_right)
        camera_up = camera_up / np.linalg.norm(camera_up)
        rotation_transform = np.zeros((4, 4))
        rotation_transform[0, :3] = camera_right
        rotation_transform[1, :3] = camera_up
        rotation_transform[2, :3] = camera_direction
        rotation_transform[-1, -1] = 1.0
        translation_transform = np.eye(4)
        translation_transform[:3, -1] = -np.array(camera_position)
        look_at_transform = np.matmul(rotation_transform, translation_transform)
        return np.linalg.inv(look_at_transform)

    def create_camera(c2w: np.ndarray, fov: float):
        from mathutils import Matrix

        camera_data = bpy.data.cameras.new("Camera")
        camera_data.angle = np.deg2rad(fov)
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        camera.matrix_world = Matrix(c2w)
        return camera

    def create_camera_look_at(camera_position, target_position, up_dir, fov: float, flip_y: bool = False):
        from mathutils import Matrix, Vector

        camera_data = bpy.data.cameras.new("Camera")
        camera_data.angle = np.deg2rad(fov)
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        direction = Vector(target_position) - Vector(camera_position)
        rot = direction.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
        if flip_y:
            rot = rot @ Matrix.Rotation(np.pi, 4, 'Z')
        loc = Matrix.Translation(Vector(camera_position))
        camera.matrix_world = loc @ rot
        return camera

    def _principled_node(material):
        return material.node_tree.nodes.get("Principled BSDF")

    def create_white_emmissive_material(strength: float, material_name: str):
        mat = bpy.data.materials.new(material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = strength
        output = nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    def create_specular_roughness_material(diffuse_color, specular_color, roughness, material_name: str):
        mat = bpy.data.materials.new(material_name)
        mat.use_nodes = True
        bsdf = _principled_node(mat)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*diffuse_color, 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = float(np.mean(specular_color))
            elif "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = float(np.mean(specular_color))
        return mat

BLENDER_BACKEND = os.getenv('BLENDER_BACKEND', 'CUDA')

from scene_config import SceneConfig, CameraConfig
try:
    from scene_mesh import generate_scene_mesh
except ImportError:
    generate_scene_mesh = None


def _as_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a Blender/imageio EXR image to [H, W, 3] float32."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] >= 3:
        return img[..., :3].astype(np.float32)
    raise ValueError(f"Unsupported image shape for RGB conversion: {img.shape}")


def _as_index(img: np.ndarray) -> np.ndarray:
    """Convert Object Index pass to int32 [H, W]. Background becomes -1 later."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    # Index passes can be saved as float EXR. Round to avoid tiny EXR errors.
    return np.rint(img).astype(np.int32)


def build_triangle_targets(
    I_direct: np.ndarray,
    I_indirect: np.ndarray,
    triangle_id_buffer: np.ndarray,
    n_triangles: int,
    min_pixels: int = 8,
    valid_pixel_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Build visible-triangle mean direct/indirect targets.

    L_direct_tri[i] = mean(I_direct[p] for pixels p that see triangle i)
    L_indirect_tri[i] = mean(I_indirect[p] for pixels p that see triangle i)
    """
    I_direct = _as_rgb(I_direct)
    I_indirect = _as_rgb(I_indirect)

    flat_ids = triangle_id_buffer.reshape(-1)
    flat_direct = I_direct.reshape(-1, 3)
    flat_indirect = I_indirect.reshape(-1, 3)

    if valid_pixel_mask is None:
        flat_valid = flat_ids >= 0
    else:
        flat_valid = valid_pixel_mask.reshape(-1) & (flat_ids >= 0)

    L_direct_tri = np.zeros((n_triangles, 3), dtype=np.float32)
    L_indirect_tri = np.zeros((n_triangles, 3), dtype=np.float32)
    counts = np.zeros((n_triangles,), dtype=np.int32)

    for tri_id in range(n_triangles):
        mask = flat_valid & (flat_ids == tri_id)
        count = int(mask.sum())
        counts[tri_id] = count
        if count > 0:
            L_direct_tri[tri_id] = flat_direct[mask].mean(axis=0)
            L_indirect_tri[tri_id] = flat_indirect[mask].mean(axis=0)

    L_total_tri = L_direct_tri + L_indirect_tri
    tri_valid_mask = counts >= min_pixels

    return {
        'L_direct_tri': L_direct_tri,
        'L_indirect_tri': L_indirect_tri,
        'L_total_tri': L_total_tri,
        'tri_visible_count': counts,
        'tri_valid_mask': tri_valid_mask.astype(np.bool_),
    }


def remove_id_boundaries(triangle_id_buffer: np.ndarray) -> np.ndarray:
    """Optional conservative mask: keep pixels whose 4-neighborhood has same ID."""
    ids = triangle_id_buffer
    valid = ids >= 0
    same = valid.copy()
    same[:-1, :] &= ids[:-1, :] == ids[1:, :]
    same[1:, :] &= ids[1:, :] == ids[:-1, :]
    same[:, :-1] &= ids[:, :-1] == ids[:, 1:]
    same[:, 1:] &= ids[:, 1:] == ids[:, :-1]
    return same


def _find_latest_file(prefix: str, ext: str = '.exr') -> str:
    files = sorted(glob.glob(prefix + '*' + ext))
    if not files:
        raise FileNotFoundError(f"No compositor output found for prefix: {prefix}*{ext}")
    return files[-1]


def _read_exr(path: str) -> np.ndarray:
    if imageio is not None:
        return imageio.v3.imread(path).astype(np.float32)

    image = bpy.data.images.load(os.path.abspath(path), check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, image.channels)
    bpy.data.images.remove(image)
    return pixels


def _configure_color_management() -> None:
    # Raw linear output is easier to use as learning target.
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0


def _configure_cycles(resolution: int, spp: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = spp
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'

    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = BLENDER_BACKEND
    scene.render.threads = 8
    scene.render.threads_mode = 'FIXED'

    # remove ambient/world contribution
    if scene.world and scene.world.node_tree:
        scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.0

    _configure_color_management()


def _enable_stage0_passes() -> None:
    view_layer = bpy.context.view_layer
    view_layer.use_pass_combined = True
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_object_index = True


def _link_if_exists(tree, render_layers, output_node, pass_name: str, slot_index: int) -> bool:
    if pass_name not in render_layers.outputs:
        print(f"[WARN] Render Layers output '{pass_name}' not found. Available: {[o.name for o in render_layers.outputs]}")
        return False
    tree.links.new(render_layers.outputs[pass_name], output_node.inputs[slot_index])
    return True


def _setup_stage0_compositor(gt_dir: str, view_prefix: str) -> dict[str, str]:
    """
    Configure compositor to dump individual EXR files for relevant passes.
    Returns dict from logical pass name to file prefix.
    """
    os.makedirs(gt_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)

    pass_to_output = {
        'combined': ('Image', f'{view_prefix}_combined_'),
        'diffuse_direct': ('DiffDir', f'{view_prefix}_diffuse_direct_'),
        'diffuse_indirect': ('DiffInd', f'{view_prefix}_diffuse_indirect_'),
        'object_index': ('IndexOB', f'{view_prefix}_object_index_'),
    }
    prefixes = {}

    for idx, (logical_name, (blender_pass_name, file_prefix)) in enumerate(pass_to_output.items()):
        out = tree.nodes.new(type='CompositorNodeOutputFile')
        out.location = (350, -160 * idx)
        out.base_path = os.path.abspath(gt_dir)
        out.file_slots[0].path = file_prefix
        out.format.file_format = 'OPEN_EXR'
        out.format.color_depth = '32'
        out.format.color_mode = 'RGBA'
        _link_if_exists(tree, render_layers, out, blender_pass_name, 0)
        prefixes[logical_name] = os.path.join(os.path.abspath(gt_dir), file_prefix)

    return prefixes


def _write_stage0_h5(
    h5_path: str,
    I_total: np.ndarray,
    I_direct: np.ndarray,
    I_indirect: np.ndarray,
    triangle_id_buffer: np.ndarray,
    c2w: np.ndarray,
    n_triangles: int,
    min_pixels: int,
    drop_id_boundaries: bool,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        h5py = None

    valid_pixel_mask = None
    if drop_id_boundaries:
        valid_pixel_mask = remove_id_boundaries(triangle_id_buffer)

    targets = build_triangle_targets(
        I_direct=I_direct,
        I_indirect=I_indirect,
        triangle_id_buffer=triangle_id_buffer,
        n_triangles=n_triangles,
        min_pixels=min_pixels,
        valid_pixel_mask=valid_pixel_mask,
    )

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    if h5py is None:
        npz_path = os.path.splitext(h5_path)[0] + '.npz'
        np.savez_compressed(
            npz_path,
            I_total=_as_rgb(I_total),
            I_direct=_as_rgb(I_direct),
            I_indirect=_as_rgb(I_indirect),
            triangle_id_buffer=triangle_id_buffer.astype(np.int32),
            valid_pixel_mask=(valid_pixel_mask if valid_pixel_mask is not None else triangle_id_buffer >= 0).astype(np.bool_),
            c2w=c2w.astype(np.float32),
            **targets,
        )
        print(f"[Stage0] h5py not available in Blender Python. Saved {npz_path} instead.")
        return

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('I_total', data=_as_rgb(I_total), compression='gzip')
        f.create_dataset('I_direct', data=_as_rgb(I_direct), compression='gzip')
        f.create_dataset('I_indirect', data=_as_rgb(I_indirect), compression='gzip')
        f.create_dataset('triangle_id_buffer', data=triangle_id_buffer.astype(np.int32), compression='gzip')
        f.create_dataset('valid_pixel_mask', data=((valid_pixel_mask if valid_pixel_mask is not None else triangle_id_buffer >= 0).astype(np.bool_)), compression='gzip')
        f.create_dataset('c2w', data=c2w.astype(np.float32))

        for key, value in targets.items():
            f.create_dataset(key, data=value, compression='gzip')

        f.attrs['n_triangles'] = int(n_triangles)
        f.attrs['min_pixels'] = int(min_pixels)
        f.attrs['drop_id_boundaries'] = bool(drop_id_boundaries)


def _split_mesh_object_to_triangle_objects(src_obj: bpy.types.Object, start_pass_index: int) -> tuple[list[bpy.types.Object], list[dict], int]:
    """
    Convert one mesh object into one Blender object per triangle.
    Returns new objects, metadata, and next pass index.
    """
    if src_obj.type != 'MESH':
        return [], [], start_pass_index

    mesh = src_obj.data
    world = src_obj.matrix_world.copy()
    mats = list(mesh.materials)
    new_objects = []
    metadata = []
    next_pass = start_pass_index

    for poly in mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) < 3:
            continue

        # Fan triangulation for non-tri polygons. Expected input is usually already triangulated.
        tri_faces = []
        if len(verts) == 3:
            tri_faces = [(verts[0], verts[1], verts[2])]
        else:
            for k in range(1, len(verts) - 1):
                tri_faces.append((verts[0], verts[k], verts[k + 1]))

        for local_fan_idx, tri in enumerate(tri_faces):
            tri_id = next_pass - 1
            coords = [world @ mesh.vertices[v_idx].co for v_idx in tri]
            new_mesh = bpy.data.meshes.new(f'{src_obj.name}_tri_mesh_{tri_id:06d}')
            new_mesh.from_pydata([tuple(v) for v in coords], [], [(0, 1, 2)])
            new_mesh.update()

            new_obj = bpy.data.objects.new(f'tri_{tri_id:06d}', new_mesh)
            bpy.context.collection.objects.link(new_obj)
            if mats:
                new_obj.data.materials.append(mats[poly.material_index])
            new_obj.pass_index = next_pass  # Object Index pass. 0 is background.
            new_objects.append(new_obj)
            metadata.append({
                'triangle_id': tri_id,
                'source_object': src_obj.name,
                'source_polygon_index': int(poly.index),
                'fan_index': int(local_fan_idx),
                'pass_index': int(next_pass),
            })
            next_pass += 1

    # Hide/delete source object after splitting so it does not render twice.
    bpy.data.objects.remove(src_obj, do_unlink=True)
    return new_objects, metadata, next_pass


def _split_scene_meshes_to_triangle_objects() -> tuple[int, list[dict]]:
    """Split all current mesh objects into triangle objects with consecutive pass_index values."""
    mesh_objs = [obj for obj in scene_meshes() if obj.type == 'MESH']
    next_pass = 1
    all_meta = []
    for obj in mesh_objs:
        _, meta, next_pass = _split_mesh_object_to_triangle_objects(obj, next_pass)
        all_meta.extend(meta)

    n_triangles = next_pass - 1
    if n_triangles > 32766:
        print(f"[WARN] n_triangles={n_triangles}. Object Index pass_index can be limited in some Blender builds.")
    print(f"[Stage0] Split mesh scene into {n_triangles} triangle objects.")
    return n_triangles, all_meta


def _assign_object_indices_without_split() -> tuple[int, list[dict]]:
    """Assign object-level pass_index. This is not true triangle ID unless every object is a triangle."""
    meta = []
    for idx, obj in enumerate([o for o in scene_meshes() if o.type == 'MESH']):
        obj.pass_index = idx + 1
        meta.append({'object_id': idx, 'object_name': obj.name, 'pass_index': idx + 1})
    print(f"[Stage0] Assigned object-level pass_index to {len(meta)} mesh objects. This is not triangle-level unless objects are triangles.")
    return len(meta), meta


def parse_scene_config(config: dict) -> SceneConfig:
    if from_dict is not None:
        return from_dict(data_class=SceneConfig, data=config, config=Config(check_types=True, strict=True))

    from scene_config import TransformConfig, MaterialConfig, ObjectConfig

    objects = {}
    for key, obj in config["objects"].items():
        material = MaterialConfig(**obj["material"])
        transform = TransformConfig(**obj["transform"])
        object_kwargs = {k: v for k, v in obj.items() if k not in ["material", "transform"]}
        objects[key] = ObjectConfig(material=material, transform=transform, **object_kwargs)

    cameras = [CameraConfig(**camera) for camera in config["cameras"]]
    return SceneConfig(
        scene_name=config["scene_name"],
        version=config["version"],
        objects=objects,
        cameras=cameras,
    )


def connect_vertex_color_to_diffuse(material) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    vcol = nodes.new(type="ShaderNodeVertexColor")
    group = nodes.get("Group")
    if group is not None and "Diffuse" in group.inputs:
        links.new(vcol.outputs["Color"], group.inputs["Diffuse"])
        return

    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None and "Base Color" in bsdf.inputs:
        links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
        return

    raise KeyError("Could not find a diffuse-compatible material input for vertex colors")


def scene_to_img(
        scene_config: SceneConfig,
        mesh_path: str,
        output_image_path: str,
        save_img: bool = False,
        resolution: int = 512,
        spp: int = 4096,
        dump_blend_file: bool = True,
        skip_rendering: bool = True,
        stage0_gt: bool = False,
        stage0_split_triangles: bool = False,
        stage0_min_pixels: int = 8,
        stage0_drop_id_boundaries: bool = False,
        stage0_flip_camera_y: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

    stage0_meta = {'n_triangles': 0, 'mapping': []}

    def setup_blender_scene(scene_config: SceneConfig, mesh_path: str) -> None:
        reset_scene()
        with stdout_redirected():
            split_mesh_path = os.path.dirname(mesh_path) + '/split'

            for obj_key, obj_config in scene_config.objects.items():
                import_3d_model(f'{split_mesh_path}/{obj_key}.obj')
                material_config = obj_config.material
                if obj_config.material.emissive[0] > 0:
                    material = create_white_emmissive_material(
                        strength=material_config.emissive[0],
                        material_name=f"{obj_key}"
                    )
                else:
                    material = create_specular_roughness_material(
                        diffuse_color=tuple(material_config.diffuse),
                        specular_color=tuple(material_config.specular),
                        roughness=material_config.roughness,
                        material_name=f"{obj_key}"
                    )
                    if material_config.rand_tri_diffuse_seed:  # Use vertex color as diffuse color when have per-triangle diffuse color
                        connect_vertex_color_to_diffuse(material)

                for obj in scene_meshes():
                    if obj.name == obj_key:
                        obj.data.materials.clear()
                        obj.data.materials.append(material)
                        obj.rotation_mode = 'XYZ'
                        obj.rotation_euler = (0.0, 0.0, 0.0)

        if stage0_gt:
            if stage0_split_triangles:
                n_triangles, mapping = _split_scene_meshes_to_triangle_objects()
            else:
                print("[WARN] --stage0_gt without --stage0_split_triangles gives object_id_buffer, not triangle_id_buffer.")
                n_triangles, mapping = _assign_object_indices_without_split()
            stage0_meta['n_triangles'] = n_triangles
            stage0_meta['mapping'] = mapping

    def render_scene(camera_config: CameraConfig, output_image_path: str, output_obj_path: str, view_idx: int) -> tuple[np.ndarray, np.ndarray]:
        camera_pos = np.array(camera_config.position)
        look_at = np.array(camera_config.look_at)
        up = np.array(camera_config.up)
        fov = camera_config.fov

        c2w = look_at_to_c2w(camera_pos, look_at, up)
        if USING_BPY_HELPER and not stage0_flip_camera_y:
            camera = create_camera(c2w, fov)
        else:
            camera = create_camera_look_at(camera_pos, look_at, up, fov, flip_y=stage0_flip_camera_y)
        bpy.context.scene.camera = camera

        _configure_cycles(resolution=resolution, spp=spp)
        if stage0_gt:
            _enable_stage0_passes()

        if skip_rendering:
            return np.zeros((resolution, resolution, 4), dtype=np.float32), c2w

        with stdout_redirected():
            temp_img_path = output_image_path.replace('.png', '.exr')

            if stage0_gt:
                gt_dir = os.path.join(os.path.dirname(output_image_path), 'stage0_gt')
                view_prefix = f'view_{view_idx:04d}'
                prefixes = _setup_stage0_compositor(gt_dir=gt_dir, view_prefix=view_prefix)
                # filepath still controls the normal Combined output.
                bpy.context.scene.render.filepath = os.path.abspath(temp_img_path)
                bpy.context.scene.frame_set(view_idx + 1)
                bpy.ops.render.render(animation=False, write_still=True)

                I_total = _as_rgb(_read_exr(_find_latest_file(prefixes['combined'])))
                I_direct = _as_rgb(_read_exr(_find_latest_file(prefixes['diffuse_direct'])))
                I_indirect = _as_rgb(_read_exr(_find_latest_file(prefixes['diffuse_indirect'])))
                index_ob = _as_index(_read_exr(_find_latest_file(prefixes['object_index'])))

                triangle_id_buffer = index_ob - 1
                triangle_id_buffer[index_ob <= 0] = -1

                h5_path = os.path.join(gt_dir, f'view_{view_idx:04d}.h5')
                _write_stage0_h5(
                    h5_path=h5_path,
                    I_total=I_total,
                    I_direct=I_direct,
                    I_indirect=I_indirect,
                    triangle_id_buffer=triangle_id_buffer,
                    c2w=c2w,
                    n_triangles=stage0_meta['n_triangles'],
                    min_pixels=stage0_min_pixels,
                    drop_id_boundaries=stage0_drop_id_boundaries,
                )
                img = np.concatenate([I_total, np.ones_like(I_total[..., :1])], axis=-1)
            else:
                bpy.context.scene.render.filepath = os.path.abspath(temp_img_path)
                bpy.ops.render.render(animation=False, write_still=True)
                img = imageio.v3.imread(temp_img_path).copy()

            if save_img:
                if imageio is not None:
                    imageio.v3.imwrite(output_image_path, (img * 255).clip(0, 255).astype(np.uint8))
                else:
                    print("[WARN] imageio is not available; skipping PNG preview save.")

        return img, c2w

    setup_blender_scene(scene_config, mesh_path)

    # Save mapping from original object/polygon to triangle_id.
    if stage0_gt:
        gt_dir = os.path.join(os.path.dirname(output_image_path), 'stage0_gt')
        os.makedirs(gt_dir, exist_ok=True)
        with open(os.path.join(gt_dir, 'triangle_id_mapping.json'), 'w') as f:
            json.dump(stage0_meta, f, indent=2)

    results = []
    for i, camera_config in tqdm(list(enumerate(scene_config.cameras))):
        img, c2w = render_scene(camera_config, f"{output_image_path}_{i}.png", f"{output_image_path}_{i}.obj", i)
        results.append((img, c2w))

    if dump_blend_file:
        bpy.ops.file.pack_all()
        save_blend_file(output_image_path + '.blend')

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render scenes using Blender')
    parser.add_argument('scene_config', type=str, help='Path to scene config JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for rendered images')
    parser.add_argument('--mesh_path', type=str,
                       help='Path to mesh file. If not provided, a temporary directory will be used',
                       default=None)
    parser.add_argument('--dump_blend', default=True, action='store_true', help='Save Blender file after rendering')
    parser.add_argument('--save_img', default=False, action='store_true', help='Save rendered images')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the rendered images')
    parser.add_argument('--spp', type=int, default=4096, help='Samples per pixel')

    # Stage-0 GT export options.
    parser.add_argument('--stage0_gt', action='store_true', help='Export Stage-0 direct/indirect/object-index H5 files')
    parser.add_argument('--stage0_split_triangles', action='store_true', help='Split every mesh polygon into triangle objects so Object Index is triangle_id_buffer')
    parser.add_argument('--stage0_min_pixels', type=int, default=8, help='Minimum visible pixels for a valid triangle target')
    parser.add_argument('--stage0_drop_id_boundaries', action='store_true', help='Ignore triangle boundary pixels when averaging triangle targets')
    parser.add_argument('--stage0_flip_camera_y', action='store_true', help='Flip Blender camera local Y/up axis for Stage-0 GT alignment')

    args = parser.parse_args()

    with open(args.scene_config) as f:
        scene_config = json.load(f)
    scene_config = parse_scene_config(scene_config)

    os.makedirs(args.output_dir, exist_ok=True)
    output_base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.scene_config))[0])

    should_render = bool(args.save_img or args.stage0_gt)

    if args.mesh_path is None:
        if generate_scene_mesh is None:
            raise ImportError("trimesh is unavailable in Blender Python. Provide --mesh_path with an existing split mesh folder.")
        print("No mesh path provided, using temporary directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
            print(f"Generating mesh in temporary path: {temp_mesh_path}")
            generate_scene_mesh(scene_config, temp_mesh_path, os.path.dirname(args.scene_config))
            scene_to_img(
                scene_config=scene_config,
                mesh_path=temp_mesh_path,
                output_image_path=output_base,
                dump_blend_file=args.dump_blend,
                save_img=args.save_img,
                resolution=args.resolution,
                spp=args.spp,
                skip_rendering=not should_render,
                stage0_gt=args.stage0_gt,
                stage0_split_triangles=args.stage0_split_triangles,
                stage0_min_pixels=args.stage0_min_pixels,
                stage0_drop_id_boundaries=args.stage0_drop_id_boundaries,
                stage0_flip_camera_y=args.stage0_flip_camera_y,
            )
    else:
        print(f"Using provided mesh path: {args.mesh_path}")
        split_mesh_path = os.path.join(os.path.dirname(args.mesh_path), "split")
        if os.path.isdir(split_mesh_path):
            print(f"Reusing existing split mesh folder: {split_mesh_path}")
        else:
            if generate_scene_mesh is None:
                raise ImportError("trimesh is unavailable in Blender Python and split mesh folder is missing.")
            generate_scene_mesh(scene_config, args.mesh_path, os.path.dirname(args.scene_config))
        scene_to_img(
            scene_config=scene_config,
            mesh_path=args.mesh_path,
            output_image_path=output_base,
            dump_blend_file=args.dump_blend,
            save_img=args.save_img,
            resolution=args.resolution,
            spp=args.spp,
            skip_rendering=not should_render,
            stage0_gt=args.stage0_gt,
            stage0_split_triangles=args.stage0_split_triangles,
            stage0_min_pixels=args.stage0_min_pixels,
            stage0_drop_id_boundaries=args.stage0_drop_id_boundaries,
            stage0_flip_camera_y=args.stage0_flip_camera_y,
        )
