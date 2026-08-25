# Warehouse danger world

`warehouse_danger_v1` composes NVIDIA's native Isaac Sim 6.0
`warehouse_with_forklifts.usd` with one local, contact-driven rack-collapse layer. No SDF,
Gazebo world, or converted mesh is involved.

The stock warehouse and Nova Carter references currently resolve from NVIDIA's
official HTTPS asset root and are cached by the existing persistent Isaac Docker
volumes. Before an offline or public recording, install the matching Isaac Sim
6.0 Local Assets Pack or use **Collect Assets**, update the two references to
relative local paths, and run the repository validation again.

The authored failure is not a timer or animation. Nova Carter must displace the
separate low-friction support through wheel-actuated contact; gravity and PhysX
contact then determine the loaded crossbeam outcome.
