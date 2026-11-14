# PE-Ledger-Solver Harness (`ple-agent-harness`)

Autonomous agent harness for **Google - Physical-Engineering's Last Exam (PLE)**.

## Architecture
1. **Living Engineering Ledger (`ledger.py`)**: Persistent structured state tracking engineering discipline, applicable codes/standards, verified physical invariants, deliverable statuses, and PE redlines across long-horizon (6-hour) tasks.
2. **Offline Engineering Reference & Physical Unit Engine (`eng_db.py`)**: Deterministic `pint` dimensional analysis, AISC steel shape tables, NEC 75°C ampacities, ASCE 7 LRFD load combinations, and geotechnical bearing capacity calculators bundled directly inside `/opt/harness`.
3. **Formal Solver Toolkit (`requirements.txt`)**: Pre-installs `z3-solver`, `scipy`, `numpy`, `sympy`, and `networkx` for exact constraint solving (3D routing, structural optimization, network flow).
4. **3-Stage Execution (`serve.py`)**:
   - **Planner**: Inspects task workspace, identifies deliverables and governing codes, and initializes the Living Engineering Ledger.
   - **Worker / Solver**: Executes engineering calculations and optimizations via Python, `pint`, `eng_lookup`, and bridged task `MCP` servers (`taskbridge.MCPClient`).
   - **Deterministic Gatekeeper + Hostile Senior PE Verifier**: Validates deliverable existence, non-zero size, Python AST syntax, and ZIP integrity, then runs an adversarial PE review loop for up to `MAX_FIX_ROUNDS = 2`.

## Model Configuration
- `model_api`: `google-cloud/claude-opus-5-5` (configured in `submission.yaml`)

## Next Step Before Final Competition Submission
Run the official PLE dev-set evaluation on a machine with Docker/WSL2 to generate `artifacts/`:
```bash
uv run python -m kit check ../ple-agent-harness
uv run python eval/run.py ../ple-agent-harness --env-file local.env
git -C ../ple-agent-harness add -A && git -C ../ple-agent-harness commit -m "harness and dev artifacts"
git -C ../ple-agent-harness push
```
