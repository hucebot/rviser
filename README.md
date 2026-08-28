# rviser

Web-based 3D visualizer (built on [viser](https://viser.studio/)) for the TIAGo robot: live camera feed, joint state, gripper trajectories, tracked objects, nav plan, orchestrator task status, and battery — all in one browser tab.

## Run it

```bash
cp .env.template .env   # set ROS_DOMAIN_ID if needed
make build
make up
```

Open **http://localhost:8080** in a browser. Useful commands:

```bash
make log      # follow container logs
make bash     # shell into the running container
make restart  # down + up
```

The container builds with `colcon build --symlink-install` and launches `rviser.launch.py`, which starts:
- `safe_teleop_node` — the viser server itself (everything below lives here)
- `server_message_bridge` — posts a plain-language status sentence to an external TV display

## Editing

`./src` is bind-mounted into the container at `/workspace/src`, and the build uses `--symlink-install`, so editing a `.py` file on the host takes effect after `make restart` — no rebuild needed. All the visualization logic lives in one file: `src/rviser/rviser/safe_teleop_node.py`.

## Where things are

| Path | What |
|---|---|
| `src/rviser/rviser/safe_teleop_node.py` | Main node: viser scene, GUI overlays, all ROS subscriptions |
| `src/rviser/scene_assets/` + `src/rviser/exported_poses/` | Static kitchen meshes (fridge, table) placed once at startup at a fixed pose exported by `scene_manager`'s manual-adjust tool. One `pose_*.json` per export; the latest timestamp wins. |
| `src/rviser/mesh_assets/` | Per-object meshes (milk, banana, ketchup, ...) shown live at the pose FoundationPose publishes on `/foundation_pose/object_marker` |
| `src/rviser/assets/` | Misc static assets (e.g. the Inria logo overlay) |
| `configs/cyclonedds.xml` | DDS peer list — add an IP here to communicate with a device or a robot |

## Adding a new mesh asset

Drop the `.obj` + `.obj.mtl` + textures in `scene_assets/` or `mesh_assets/<name>/`. **Watch out for scanned/photogrammetry meshes with thousands of `newmtl` entries in the `.mtl`** (Meshlab often exports one material per face) — trimesh hangs or OOMs trying to resolve them. The code auto-detects this (`_mesh_has_excessive_materials`, threshold 50) and falls back to a flat grey material rather than hanging, but it's better to fix the `.mtl` itself: collapse it down to one `newmtl` per unique texture (see how `ketchup.obj.mtl` / the `scene_assets/*.mtl` files were fixed — same texture, just drop the per-face `Kd` tint and point every `usemtl` at the single consolidated material).