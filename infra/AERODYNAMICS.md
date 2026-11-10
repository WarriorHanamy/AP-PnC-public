# Quadrotor plant

This line models a multirotor. Collective thrust is along body `+Z` (FLU), and
the planner's differential flatness aligns that axis with specific force. There
is no wing-force model, no transition or cruise aerodynamic mode, and no choice
among `lyu`, `phi`, or other aero datasets.

The NMPC state is position, velocity, body-rate command, attitude, and specific
thrust. Regenerating the solver (`ap-pnc gen-nmpc-lib`) is required after this
model change because the generated bundle is statically linked.

Post-CIRI reinforcement learning and the radar-depth YOPO baseline are
documented in [LEARN.md](LEARN.md). CIRI reconstructions are not in this tree.
