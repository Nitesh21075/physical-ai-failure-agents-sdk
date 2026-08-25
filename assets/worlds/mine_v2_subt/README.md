# mine_v2_subt

`mine_v2_subt` is the visually rich perception counterpart to the deterministic
`mine_v1` regression world. It composes a pinned, converted LTU DARPA SubT cave
mesh with the existing Nova Carter definition and unchanged roof-support
failure asset. It never modifies or replaces `mine_v1`.

The imported cave mesh is visual-only in this first prototype. A simple static
floor collider bounds the verified roof-support route; claiming traversal or
contact elsewhere in the cave requires additional collision authoring and a
real Isaac smoke test. Lighting and camera poses are authored for RTX rather
than inherited from Gazebo.

## Rebuild the converted source asset

Checkout the pinned source commit, then run the converter in Isaac Sim 6.0.1:

```bash
docker run --rm --gpus all --network host --user 0:0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/tmp \
  -v "$PWD:/workspace/project" -v "/path/to/darpa_subt_worlds:/source:ro" \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc 'cd /workspace/project && /isaac-sim/python.sh scripts/import_ltu_subt_cave.py \
    --source-repository /source \
    --output /workspace/project/assets/worlds/mine_v2_subt/vendor/ltu_darpa_subt/cave_world.usdc'
```

The converter copies referenced textures beside the output and writes a
provenance JSON file. Review generated changes and hashes before committing.

## Bounded experiment

Use the existing runner with an explicit stage:

```bash
/isaac-sim/python.sh scripts/run_mine_rover_experiment.py \
  --runs-dir /workspace/project/runs/isaac-mine \
  --stage /workspace/project/assets/worlds/mine_v2_subt/mine_world.usda \
  --failure-zone roof_support --linear-velocity-mps 0.25 --control-steps 180
```

The bounded route passed real Nova Carter/PhysX and RTX checks on 2026-08-24.
The Agents SDK researcher intentionally remains on deterministic `mine_v1`;
`mine_v2_subt` is currently an explicit-stage perception prototype, and no
claim is made about collision or traversability outside its proxy floor.
