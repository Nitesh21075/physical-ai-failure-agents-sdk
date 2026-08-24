# Mine Rover Camera Experiment

`scripts/run_mine_rover_experiment.py` is the bounded Isaac Sim worker for a
mine-world run. It commands the actual Nova Carter wheel articulation, captures
three synchronized RTX RGB views, and records a quality-gated Reactor seed
handoff. It does not move the rover chassis directly.

## Non-destructive boundary

`assets/worlds/mine_v1/mine_world.usda` and every USD it composes are campaign
inputs. For each run the worker creates:

```text
runs/isaac-mine/<run-id>/mine_rover_derived_entry.usda  # source-world overlay
runs/isaac-mine/<run-id>/mine_rover_session.usda        # session-only edits
runs/isaac-mine/<run-id>/camera/ego/rgb_*.png           # rover-oriented view
runs/isaac-mine/<run-id>/camera/tracking/rgb_*.png      # smoothed chase view
runs/isaac-mine/<run-id>/camera/witness/rgb_*.png       # fixed failure-zone view
runs/isaac-mine/<run-id>/camera/camera_poses.jsonl      # camera provenance
runs/isaac-mine/<run-id>/reactor_seed.json              # visual-world seed
runs/isaac-mine/<run-id>/summary.json                   # pose/physics provenance
```

The derived entry layer selects Nova Carter's `Config=Full_Merged` and
`Physics=physx` variants before the remote asset is composed. These are
the lightweight, physics-capable settings; the worker never selects the
`No_Physics` variant. Every wheel name is discovered from the live articulation
and the requested linear/angular command is converted to bounded differential
wheel velocities.

## Camera roles and evidence gate

The run-owned session layer creates `/World/Sensors/RoverEgoCamera` and
`/World/Sensors/RoverTrackingCamera`, and repairs the existing
`/World/Sensors/RoofSupportCamera`. The ego and tracking poses are recomputed
from the measured articulation pose after physics updates. The tracking pose is
deterministically smoothed and has no collision, rigid body, or physical effect.
The fixed witness view remains pointed at the support/beam bay.

The tracking stream is the Reactor seed. Pair preparation rejects it unless
every tracking frame passes measured contrast/range/gradient checks, the
tracking view contains the semantically labelled roof support, the witness view
contains the semantically labelled beam, and the tracking camera remains within
the bounded chase distance from the measured rover articulation. Isaac 6.0.1
does not return the authored class label for NVIDIA's instanced Nova Carter
visuals, so rover framing uses the measured articulation/camera geometry and
records that limitation explicitly. No placeholder frame is substituted.

## Run

From this checkout, create persistent, disposable Isaac caches once. They are
Docker volumes, not files in the world directory.

```bash
docker volume create mine-rover-isaac-cache
docker volume create mine-rover-omniverse-data
docker run --rm --gpus all --network host --user 0:0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/tmp \
  -v mine-rover-isaac-cache:/tmp/.cache \
  -v mine-rover-omniverse-data:/tmp/.nvidia-omniverse \
  -v "$PWD:/workspace/project" \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc 'cd /workspace/project && /isaac-sim/python.sh scripts/run_mine_rover_experiment.py \
    --runs-dir /workspace/project/runs/isaac-mine --run-id mine-rover-001 \
    --failure-zone roof_support --linear-velocity-mps 0.25 --control-steps 180'
```

Use `--disable-camera` only for an articulation/PhysX diagnostic. It produces
no RGB image and therefore intentionally produces no Reactor seed.

The first Nova Carter load can take several minutes because the reference asset
is resolved from NVIDIA's content service. Keep the two cache volumes between
runs; remove them only when intentionally forcing another cold-load experiment.

## Reactor comparison boundary

`reactor_seed.json` supplies the first accepted tracking-camera RGB frame, its
camera role/prim and quality evidence, the exact source/session
USD provenance, command, and before/after rover poses to a Reactor workflow.
It is an image-conditioning handoff, not a claim that a generated visual world
is simulator ground truth. Physics outcomes remain measured in Isaac through
the wheel command, contacts, and body poses; Reactor is the comparison branch.

The RTX implementation follows NVIDIA's current experimental RTX camera API,
not the deprecated `isaacsim.sensors.camera` wrapper:

- [Isaac Sim RTX camera migration](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/sensors_camera_to_experimental_rtx.html)
- [NVIDIA Nova Carter asset variants](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/nova_carter_landing_page.html)
