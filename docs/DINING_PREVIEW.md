# Dining scene preview (Colab only)

The scene adds original primitive walnut planks, woven-look placemats, two guest
settings, brass cutlery, folded napkins, a vase, candles, chairs, and wall art.
Decorations are static and non-colliding; they are visual context, not additional
manipulation targets. Existing robot bases, task objects, and tabletop collision
geometry are unchanged. No third-party decorative assets are used.

Run this cell in the existing Colab notebook (CPU is sufficient for a still):

```python
%cd /content/recovervla
!git pull --ff-only
!apt-get update -qq
!apt-get install -y -qq libosmesa6
!pip install -q mujoco==3.3.7 numpy pillow
from pathlib import Path
if not Path('artifacts/so101/manifest.json').exists():
    !python scripts/fetch_assets.py
!MUJOCO_GL=osmesa python scripts/render_scene.py --seed 0
from IPython.display import display, Image
display(Image(filename='artifacts/preview/hero.png'))
display(Image(filename='artifacts/preview/overhead.png'))
```

The script also saves the scene and both wrist-camera views. These are static
initial-state renders, not evidence of collision-free motion or learned-policy
success. Remote compilation and all five camera renders passed in Colab on
2026-09-16 with seed 0 (MuJoCo 3.3.7, CPU/OSMesa). Physical rollout checks are
still required before collecting training data. No local execution.
