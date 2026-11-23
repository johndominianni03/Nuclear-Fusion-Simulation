"""Tier 3.5: rendered 3D animation of the fusion-product cycle.

    viz.scene   SceneConfig (the code inputs) and load_scene -> Scene (stored arrays only)
    viz.rerun   re-simulate a scene's markers with physics overrides (cached by hash)
    viz.render  one frame at physical time t
    viz.make    CLI: frames (cached, resumable) -> GIF + MP4

Nothing here changes a physics module; every drawn point is a stored array element.
"""
