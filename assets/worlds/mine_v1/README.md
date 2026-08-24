# mine_v1: persistent mine failure-research world

`mine_world.usda` is the directly openable entry stage. It composes the static drift, materials, lighting, physics defaults, three hazard zones, and NVIDIA's articulated Nova Carter rover. Mine-authored references are relative; the selected rover reference is the documented NVIDIA Isaac 6.0 cloud asset URL, with a local proxy fallback variant.

## Layout

```text
mine_world.usda                 composed entry stage
mine_base.usda                  original static mine shell and props
sections/extended_drift.usda    connected 32m static drift extension
layers/                         physics, materials, lighting sublayers
hazards/                        authored roof-support, rockfall, and debris components
robots/inspection_robot.usda    Nova Carter reference + local proxy fallback variant
manifest.json                   stable prim registry for workers/tools
```

The installed Isaac Sim 6.0.1 image was inspected before authoring and does not include a local portable robot pack. The default robot is therefore a network-resolved NVIDIA Isaac 6.0 Nova Carter asset. It has PhysX articulation, wheels, cameras, IMUs, and lidars; its 3D-content-sharing license remains an external dependency. `/World/Robot` stays stable, and `robot_model=proxy_fallback` preserves an entirely local fallback if the asset root is unavailable.

## Physics model

Static scenery (`/World/Environment`) uses static collision geometry only. The beam, support, loose rocks, debris, and robot proxy are small independent rigid-body assemblies with collision geometry and explicit masses. This keeps the experiment count low and behavior inspectable.

The original `Zone_RoofSupport`, `Zone_Rockfall`, and `Zone_Debris` paths are
unchanged for compatibility. The connected extension begins beyond the former
end wall (the wall was retained and moved to the new far drift face, which was
necessary to make the route physically continuous). It contains the new
explicitly named rover zones:

```text
existing mine -> RoofSupportZone (x=24) -> RockfallZone (x=35) -> DebrisCascadeZone (x=43..45)
```

The primary causal demonstration is the extended roof-support zone:

```text
rover chassis contacts SupportPrimary/PushFace
  -> SupportPrimary (separate 8 kg low-friction body) moves laterally
  -> BeamPrimary loses its only collision support
  -> contact support disappears
  -> PhysX gravity drops BeamPrimary
```

There is no timer, scripted collapse event, or scripted switch from kinematic
to dynamic for `BeamPrimary`. `SupportSecondary` is visibly shorter, has no
collider, and cannot secretly hold the beam after the primary support moves.

## Rover-caused failures

The default rover is NVIDIA Nova Carter: a real articulated wheeled robot, not
a fake event trigger. The retained mine runner commands its wheel joints and
captures `/World/Robot/VLACamera`. It does not teleport the chassis during an
experiment or invoke a `collapse` command.

- **Roof support:** approach the negative-Y side of
  `SupportPrimary/PushFace`, then drive in positive Y. The support is a
  distinct 22 kg body; moving it clear of the beam removes the only intended
  support and gravity causes the failure.
- **Rockfall:** approach the negative-Y side of
  `RetainingObject/PushFace`, then drive in positive Y. The dynamic retainer
  moves clear of the down-slope route, allowing the loose rocks to roll from
  the separate tilted ledge.
- **Debris:** drive the chassis into `Debris01` or `Debris02`; both are
  independent rigid bodies and require no special trigger.

These interaction recipes are registered in `manifest.json`. Only roof support
is currently exposed as an executable Agents SDK tool. Rockfall and debris are
authored world capabilities, not executable claims. The roof-support runner
uses normal wheel motion and records body poses and contact-derived outcomes.

## New stable failure-zone paths

```text
/World/FailureZones/RoofSupportZone
  /BeamPrimary
  /SupportPrimary
  /SupportSecondary
  /InteractionTarget
  /InspectionMarker

/World/FailureZones/RockfallZone
  /Rock01
  /Rock02
  /RetainingBlock
  /InteractionTarget
  /InspectionMarker

/World/FailureZones/DebrisCascadeZone
  /Debris01
  /Debris02
  /Debris03
  /InteractionTarget
```

`/World/Sensors/RoofSupportCamera`, `/World/Sensors/RockfallCamera`, and
`/World/Sensors/DebrisCamera` retain the active rover, interaction object, and
failure area in dedicated reference views. The existing research and overview
cameras are untouched.

## Parameterization and reproducible experiments

The manifest's `failure_zones` registry maps every experiment-facing path.
An experiment/session layer can override selected authored properties without
changing the base world. The current agent tool exposes only rover linear
velocity, control-step count, and seed. It does not advertise support offset,
mass, friction, arbitrary prim paths, or world editing because those per-run
controls have not been accepted. There is no arbitrary script-execution field.

## Validate in the AWS Isaac container

Run from the repository checkout that contains this world:

```bash
docker run --rm --gpus all --network host --user 0:0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/tmp -e OMNI_KIT_ALLOW_ROOT=1 \
  -v mine-rover-isaac-cache:/tmp/.cache \
  -v mine-rover-omniverse-data:/tmp/.nvidia-omniverse \
  -v "$PWD:/workspace/project" \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc 'cd /workspace/project && /isaac-sim/python.sh scripts/validate_mine_world.py \
    --smoke-test --skip-render \
    --record-dir /workspace/project/runs/mine-world-validation'
```

The command opens the real composed stage, verifies the stable paths and physics
APIs, steps PhysX, and runs the bounded roof-support rover-contact smoke test.
The runner places the existing rover chassis at an ephemeral start pose and
applies bounded forward velocity; only real chassis/support/beam interaction can
cause the observed fall. It never writes the source world. Use the mine-runner
camera frames from an indexed run as Reactor evidence.

## Open interactively

Start Isaac Sim 6.0.1 with the project directory mounted at `/workspace/project` (the same mount as the validation command), then choose:

```text
File -> Open -> /workspace/project/assets/worlds/mine_v1/mine_world.usda
```

For an existing AWS container whose checkout is mounted at `/workspace/project`, the in-container path is the same. Do not put host `/home/ubuntu/...` paths in the USD.

## Experiment composition policy

Treat all files in this directory as immutable campaign inputs. A worker should load `mine_world.usda`, then author its changes in a stronger session or derived layer. Safe experiment operations are enumerated in `manifest.json`; they include moving `SupportPrimary`, setting its kinematic state for deterministic displacement, altering listed rigid-body properties, repositioning the robot, and adding an approved object. Do not flatten a composed experiment except when exporting a snapshot.
