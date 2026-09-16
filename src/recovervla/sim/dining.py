"""Original primitive dining-room assets; no external textures or mesh licenses.

Decor is static and non-colliding, outside the central manipulation area.
The tabletop and task objects retain their physical collision geometry.
"""
import math
import xml.etree.ElementTree as ET


def decorate(root, world, assets):
    def material(name, rgb, **kw):
        ET.SubElement(assets, 'material', name=name, rgba=rgb + ' 1', **kw)

    for name, rgb, spec, shine in [
        ('walnut', '.24 .115 .055', '.25', '.3'),
        ('wood_edge', '.12 .055 .026', '.3', '.4'),
        ('porcelain', '.92 .89 .79', '.5', '.6'),
        ('teal', '.055 .23 .22', '.4', '.5'),
        ('brass', '.7 .46 .17', '.8', '.8'),
        ('linen', '.56 .36 .22', '.05', '.05'),
        ('cream', '.72 .68 .57', '.05', '.05'),
        ('leaf', '.16 .3 .14', '.1', '.1'),
        ('charcoal', '.06 .075 .07', '.2', '.2')]:
        material(name, rgb, specular=spec, shininess=shine)

    def shape(name, kind, size, pos, mat, **kw):
        return ET.SubElement(world, 'geom', name='decor_' + name, type=kind,
                             size=' '.join(map(str, size)), pos=' '.join(map(str, pos)),
                             material=mat, contype='0', conaffinity='0', **kw)

    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='1200')
    ET.SubElement(visual, 'quality', shadowsize='4096', offsamples='4')
    ET.SubElement(visual, 'headlight', ambient='.2 .2 .2', diffuse='.3 .3 .3', specular='.15 .15 .15')
    ET.SubElement(visual, 'map', znear='.01', zfar='20')
    ET.SubElement(world, 'light', name='warm_key', pos='-.6 -.3 1.4', dir='.3 .2 -1',
                  diffuse='.85 .74 .59', specular='.4 .35 .25', castshadow='true')
    ET.SubElement(world, 'light', name='window_fill', pos='.9 .3 1', dir='-.7 -.2 -1',
                  diffuse='.45 .56 .68', castshadow='false')
    ET.SubElement(world, 'camera', name='hero', pos='1.05 -1.35 1.02',
                  xyaxes='.79 .61 0 -.32 .42 .85', fovy='43')
    ET.SubElement(world, 'camera', name='overhead', pos='0 0 1.65', xyaxes='1 0 0 0 1 0', fovy='48')
    ET.SubElement(world, 'camera', name='workspace', pos='0 -.58 .52',
                  xyaxes='1 0 0 0 .55 .83', fovy='52')

    # Visual planks sit just above the unchanged collision tabletop (z=.025).
    for i in range(9):
        x = -.4 + i * .1
        shape(f'plank_{i}', 'box', (.0495, .35, .0004), (x, 0, .0254), 'walnut')
        for j in range(4):
            shape(f'grain_{i}_{j}', 'box', (.00025, .31, .00008),
                  (x - .033 + j * .019, 0, .02588), 'wood_edge')
    shape('apron', 'box', (.445, .345, .035), (0, 0, -.042), 'wood_edge')
    for x in (-.38, .38):
        for y in (-.28, .28):
            shape(f'leg_{x}_{y}', 'box', (.025, .025, .30), (x, y, -.36), 'walnut')
            shape(f'foot_{x}_{y}', 'box', (.0255, .0255, .035), (x, y, -.625), 'brass')

    # Two guest settings at the front edge, away from the task drawer.
    for n, x in enumerate((-.25, .25)):
        y = -.225
        shape(f'mat_{n}', 'box', (.105, .083, .0007), (x, y, .026), 'linen')
        for j in range(17):
            shape(f'weave_{n}_{j}', 'box', (.1, .00035, .00015),
                  (x, y - .075 + j * .009, .02685), 'cream')
        shape(f'charger_{n}', 'cylinder', (.063, .0018), (x, y, .029), 'brass')
        shape(f'dinnerplate_{n}', 'cylinder', (.057, .002), (x, y, .032), 'porcelain')
        shape(f'platewell_{n}', 'cylinder', (.043, .0005), (x, y, .0345), 'teal')
        shape(f'napkin_{n}', 'box', (.022, .038, .002), (x, y, .037), 'cream', euler='0 0 .2')
        shape(f'napkinband_{n}', 'box', (.023, .006, .0005), (x, y, .0395), 'brass', euler='0 0 .2')
        for sign in (-1, 1):
            sx = x + sign * .079
            shape(f'cutlery_{n}_{sign}', 'capsule', (.002, .025), (sx, y, .03),
                  'brass', euler='1.570796 0 0')
            if sign == -1:
                for t in range(4):
                    shape(f'tine_{n}_{t}', 'box', (.0007, .009, .0008),
                          (sx - .003 + t * .002, y + .029, .030), 'brass')
            else:
                shape(f'blade_{n}', 'box', (.003, .018, .001), (sx, y + .024, .03), 'brass')

    # Back-edge centerpiece: a ceramic vase with stylized eucalyptus stems.
    shape('runner', 'box', (.11, .046, .0008), (0, .285, .026), 'cream')
    shape('vase_base', 'cylinder', (.022, .019), (0, .285, .046), 'teal')
    shape('vase_neck', 'cylinder', (.012, .015), (0, .285, .075), 'teal')
    for i in range(5):
        angle = i * 2 * math.pi / 5
        dx, dy = .022 * math.cos(angle), .022 * math.sin(angle)
        shape(f'stem_{i}', 'capsule', (.001,), (0, 0, 0), 'leaf',
              fromto=f'0 .285 .08 {dx} { .285 + dy} .18')
        for j in range(3):
            shape(f'leaf_{i}_{j}', 'ellipsoid', (.013, .005, .002),
                  (dx * (j + 1) / 3, .285 + dy * (j + 1) / 3, .115 + j * .026),
                  'leaf', euler=f'0 .4 {angle}')
    for i, x in enumerate((-.075, .075)):
        shape(f'candleholder_{i}', 'cylinder', (.014, .005), (x, .285, .031), 'brass')
        shape(f'candle_{i}', 'cylinder', (.009, .023), (x, .285, .057), 'porcelain')

    shape('floor', 'plane', (3, 3, .02), (0, 0, -.67), 'cream')
    shape('wall', 'box', (2, .025, 1.1), (0, 1.0, .43), 'cream')
    for i in range(17):
        shape(f'wall_slat_{i}', 'box', (.016, .018, .5), (-1.2 + i * .15, .962, -.15), 'walnut')
    shape('wall_rail', 'box', (2, .025, .015), (0, .95, .36), 'wood_edge')
    shape('art_frame', 'box', (.23, .02, .18), (0, .94, .73), 'brass')
    shape('art_canvas', 'box', (.216, .006, .166), (0, .915, .73), 'teal')
    shape('art_sun', 'ellipsoid', (.075, .008, .075), (.07, .905, .78), 'linen')
    for n, x in enumerate((-.25, .25)):
        shape(f'chair_seat_{n}', 'box', (.12, .115, .023), (x, -.55, -.23), 'teal')
        shape(f'chair_back_{n}', 'box', (.12, .021, .14), (x, -.66, -.06), 'teal')
        for dx in (-.09, .09):
            for y in (-.63, -.47):
                shape(f'chair_leg_{n}_{dx}_{y}', 'capsule', (.011, .205), (x + dx, y, -.455), 'walnut')
