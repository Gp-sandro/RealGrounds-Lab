"""RealGrounds (voice fork): photos of a room -> a measured room an assistant can reason about.

Modules: depth (MoGe-2: metric depth + lens), segformer (floor / wall / ceiling), detect (Grounding DINO + SAM),
room (3D boxes, relations, what rests on what), support (the learned common sense), physics (MuJoCo checks),
model_cache (reuse the networks' outputs), agent/ (floor planner, skills, commands).
"""
