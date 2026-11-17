#!/usr/bin/env python3
"""Builds monster.glb, items.glb and environment.glb for the Backrooms game.
Run:  python3 tools/make_models.py      (needs only numpy)
Open any .glb in Blender via File > Import > glTF to edit it, then re-export."""
import numpy as np, json, struct, math, os
A = np.array
def nz(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
def grid(nu, nv):
    j, i = np.meshgrid(np.arange(nv - 1), np.arange(nu - 1), indexing='ij')
    a = (j * nu + i).ravel(); b = a + 1; c = a + nu; d = c + 1
    return np.stack([a, c, b, b, c, d], -1).reshape(-1, 3)
def Rx(a): c, s = math.cos(a), math.sin(a); return A([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = math.cos(a), math.sin(a); return A([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = math.cos(a), math.sin(a); return A([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ---------- primitives: return (P, N, UV, I) ----------
def ellipsoid(c, r, nu=28, nv=18, phi=(0, 2 * math.pi), th=(0, math.pi)):
    U, V = np.meshgrid(np.linspace(*phi, nu), np.linspace(*th, nv))
    d = np.stack([-np.cos(U) * np.sin(V), np.cos(V), np.sin(U) * np.sin(V)], -1).reshape(-1, 3)
    uv = np.stack([(U - phi[0]) / (phi[1] - phi[0]), 1 - (V - th[0]) / (th[1] - th[0])], -1).reshape(-1, 2)
    return A(c) + d * A(r), nz(d / A(r)), uv, grid(nu, nv)
def ball(c, r): return ellipsoid(c, (r, r, r), 16, 10)
def pathtube(pts, rad, n=14):
    pts = A(pts, float); m = len(pts)
    rad = np.full(m, rad, float) if np.isscalar(rad) else A(rad, float)
    T = [nz(pts[min(k + 1, m - 1)] - pts[max(k - 1, 0)]) for k in range(m)]
    a = A([0, 0, 1.]) if abs(T[0][2]) < .9 else A([1., 0, 0])
    u = nz(np.cross(T[0], a)); P = []; N = []; ang = np.linspace(0, 2 * np.pi, n)
    for k in range(m):
        t = T[k]; u = nz(u - t * np.dot(u, t)); w = np.cross(t, u)
        d = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
        P.append(pts[k] + d * rad[k]); N.append(d)
    return np.concatenate(P), np.concatenate(N), np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m)), -1).reshape(-1, 2), grid(n, m)
def tube(p0, p1, r0, r1, n=14): return pathtube([p0, p1], [r0, r1], n)
def box(c, s, R=None):
    h = A(s, float) / 2; P = []; N = []; I = []; UV = []
    for ax in range(3):
        for sg in (-1, 1):
            n = np.zeros(3); n[ax] = sg; u = np.zeros(3); u[(ax + 1) % 3] = 1; v = np.zeros(3); v[(ax + 2) % 3] = 1
            f = n * h; q = A([f - u * h - v * h, f + u * h - v * h, f + u * h + v * h, f - u * h + v * h])
            b = 4 * len(P); tri = [[0, 1, 2], [0, 2, 3]]
            if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), n) < 0: tri = [[0, 2, 1], [0, 3, 2]]
            P.append(q); N.append(np.tile(n, (4, 1))); I.append(A(tri) + b); UV.append(A([[0, 0], [1, 0], [1, 1], [0, 1]]))
    P = np.concatenate(P); N = np.concatenate(N); I = np.concatenate(I)
    if R is not None: P = P @ R.T; N = N @ R.T
    return P + A(c, float), N, np.concatenate(UV), I
def lathe(prof, n=32):
    """profile list of (radius, y) going bottom-center -> outside -> up -> top-center"""
    prof = A(prof, float); m = len(prof); ang = np.linspace(0, 2 * np.pi, n); P = []; N = []
    for k in range(m):
        r, y = prof[k]; a, b = prof[max(k - 1, 0)], prof[min(k + 1, m - 1)]
        nr, ny = b[1] - a[1], -(b[0] - a[0]); l = math.hypot(nr, ny) or 1; nr /= l; ny /= l
        for t in ang: P.append([r * math.cos(t), y, r * math.sin(t)]); N.append([nr * math.cos(t), ny, nr * math.sin(t)])
    return A(P), A(N), np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m)), -1).reshape(-1, 2), grid(n, m)

# ---------- scene graph + GLB writer ----------
class Geo:
    def __init__(s): s.l = []; s.n = 0
    def add(s, P, N, U, I):
        f = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]); vn = N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]]
        if (f * vn).sum() < 0: I = I[:, [0, 2, 1]]
        s.l.append((P, N, U, I + s.n)); s.n += len(P)
    def arr(s): return [np.concatenate([x[k] for x in s.l]) for k in range(4)]
class Node:
    def __init__(s, name, t=(0, 0, 0), r=(0, 0, 0)): s.name = name; s.t = t; s.r = r; s.g = {}; s.c = []
    def add(s, mat, prim, M=None):
        P, N, U, I = prim
        if M is not None: R, T = M; P = P @ R.T + A(T); N = nz(N @ R.T)
        s.g.setdefault(mat, Geo()).add(P, N, U, I); return s
    def child(s, n): s.c.append(n); return n
def quat(x, y, z):
    c1, c2, c3 = math.cos(x / 2), math.cos(y / 2), math.cos(z / 2); s1, s2, s3 = math.sin(x / 2), math.sin(y / 2), math.sin(z / 2)
    return [s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3]
def col(h): return [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255]
MATS = {}
def mat(name, c, rough=.8, metal=0, em=None):
    MATS[name] = dict(c=c, r=rough, m=metal, e=em)
def write_glb(path, roots):
    buf = bytearray(); views = []; accs = []; meshes = []; nodes = []; names = list(MATS); gm = []
    for k in names:
        m = MATS[k]; d = {'name': k, 'pbrMetallicRoughness': {'baseColorFactor': col(m['c']) + [1], 'metallicFactor': m['m'], 'roughnessFactor': m['r']}}
        if m['e']: d['emissiveFactor'] = [x * m['e'][1] for x in col(m['e'][0])]
        gm.append(d)
    def acc(arr, ty, tg, ct):
        while len(buf) % 4: buf.append(0)
        views.append({'buffer': 0, 'byteOffset': len(buf), 'byteLength': arr.nbytes, 'target': tg}); buf.extend(arr.tobytes())
        a = {'bufferView': len(views) - 1, 'componentType': ct, 'count': len(arr), 'type': ty}
        if ty == 'VEC3' and tg == 34962 and ct == 5126 and acc.pos:
            a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
        accs.append(a); return len(accs) - 1
    def build(n):
        d = {'name': n.name}; nodes.append(d); idx = len(nodes) - 1
        if any(n.t): d['translation'] = list(map(float, n.t))
        if any(n.r): d['rotation'] = quat(*n.r)
        if n.g:
            prims = []
            for mn, g in n.g.items():
                P, N, U, I = g.arr(); acc.pos = True
                pa = acc(P.astype('f4'), 'VEC3', 34962, 5126); acc.pos = False
                na = acc(N.astype('f4'), 'VEC3', 34962, 5126); ua = acc(U.astype('f4'), 'VEC2', 34962, 5126)
                ia = acc(I.astype('u4').ravel(), 'SCALAR', 34963, 5125)
                prims.append({'attributes': {'POSITION': pa, 'NORMAL': na, 'TEXCOORD_0': ua}, 'indices': ia, 'material': names.index(mn)})
            meshes.append({'name': n.name, 'primitives': prims}); d['mesh'] = len(meshes) - 1
        if n.c: d['children'] = [build(c) for c in n.c]
        return idx
    acc.pos = False
    top = [build(r) for r in roots]
    js = {'asset': {'version': '2.0', 'generator': 'make_models.py'}, 'scene': 0, 'scenes': [{'nodes': top}], 'nodes': nodes,
          'meshes': meshes, 'materials': gm, 'accessors': accs, 'bufferViews': views, 'buffers': [{'byteLength': len(buf)}]}
    j = json.dumps(js).encode(); j += b' ' * (-len(j) % 4)
    while len(buf) % 4: buf.append(0)
    out = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(j) + 8 + len(buf)) + struct.pack('<I4s', len(j), b'JSON') + j + struct.pack('<I4s', len(buf), b'BIN\0') + bytes(buf)
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, 'wb').write(out)
    print(path, round(len(out) / 1024), 'KB,', len(nodes), 'nodes,', sum(len(m['primitives']) for m in meshes), 'primitives')

# ================= MONSTER =================
def monster():
    mat('skin', 0x5a4a42, .55); mat('flesh', 0x2a1c1c, .7); mat('Face', 0xffffff, .6)
    roots = []
    ab = Node('Abdomen', (0, 1.35, 0)); ab.add('flesh', ellipsoid((0, 0, 0), (.24, .5, .17))); roots.append(ab)
    ch = Node('Chest', (0, 2.0, 0)); ch.add('flesh', ellipsoid((0, 0, 0), (.3, .65, .2), 36, 24))
    for k in range(5):  # rib bands + collarbones
        y = .4 - k * .15; s = math.sqrt(max(0.05, 1 - (y / .65) ** 2))
        ch.add('flesh', ellipsoid((0, y, .01), (.3 * s + .012, .018, .2 * s + .012), 28, 6))
    for sx in (-1, 1): ch.add('flesh', tube((0, .52, .13), (sx * .3, .5, .04), .02, .018, 8))
    roots.append(ch)
    nk = Node('Neck', (0, 2.65, 0)); nk.add('skin', tube((0, -.25, 0), (0, .25, 0), .07, .055, 16))
    for sx in (-1, 1): nk.add('skin', tube((sx * .04, -.22, .04), (sx * .03, .2, .045), .014, .012, 8))
    roots.append(nk)
    hd = Node('Head', (0, 3.0, 0)); hd.add('skin', ellipsoid((0, 0, 0), (.3, .38, .3), 40, 28))
    hd.add('skin', ellipsoid((0, -.3, .06), (.2, .13, .2), 24, 14))
    for sx in (-1, 1): hd.add('skin', ellipsoid((sx * .3, 0, -.02), (.035, .1, .06), 12, 10))
    hd.child(Node('Face')).add('Face', ellipsoid((0, 0, 0), (.3 * 1.012, .38 * 1.012, .3 * 1.012), 40, 30, (math.pi / 2 - .8, math.pi / 2 + .8), (math.pi / 2 - .95, math.pi / 2 + .95)))
    roots.append(hd)
    for side, nm in ((-1, 'L'), (1, 'R')):  # arms: shoulder, elbow, long fingers
        arm = Node('Arm' + nm, (side * .45, 2.45, 0)); arm.add('skin', ball((0, 0, 0), .12)); arm.add('skin', tube((0, 0, 0), (0, -.95, 0), .1, .075, 18))
        fa = arm.child(Node('Forearm' + nm, (0, -.95, 0))); fa.add('skin', ball((0, 0, 0), .085)); fa.add('skin', tube((0, 0, 0), (0, -.9, 0), .075, .05, 18))
        fa.add('skin', ellipsoid((0, -.97, 0), (.07, .09, .03), 20, 12))
        for k, (x, L) in enumerate(((-.055, .21), (-.02, .24), (.015, .23), (.05, .2))):
            pts = [A([x, -1.03, 0.])]
            for l, a in zip((L, L * .85, L * .65), (.12, .5, .9)): pts.append(pts[-1] + A([0, -math.cos(a), math.sin(a)]) * l)
            fa.add('skin', pathtube(pts, [.02, .017, .013, .009], 10)); fa.add('skin', ball(pts[-1], .011))
        th = [A([-side * .07, -.94, .01])]
        for l in (.16, .13, .1): th.append(th[-1] + nz(A([-side * .5, -.75, .35])) * l)
        fa.add('skin', pathtube(th, [.022, .018, .013, .01], 10))
        roots.append(arm)
    for side, nm in ((-1, 'L'), (1, 'R')):  # legs
        lg = Node('Leg' + nm, (side * .18, 1.3, 0)); lg.add('skin', ball((0, 0, 0), .15)); lg.add('skin', tube((0, 0, 0), (0, -.7, 0), .13, .09, 18))
        sh = lg.child(Node('Shin' + nm, (0, -.7, 0))); sh.add('skin', ball((0, 0, 0), .09)); sh.add('skin', tube((0, 0, 0), (0, -.66, 0), .085, .05, 16))
        sh.add('skin', ellipsoid((0, -.7, .1), (.08, .05, .19), 18, 10)); roots.append(lg)
    sp = Node('Spider')
    for i in range(8):
        ry = (1 if i < 4 else -1) * (.35 + (i % 4) * .5) + (0 if i < 4 else math.pi)
        leg = sp.child(Node('SpiderLeg%d' % i, (0, 1.3, 0), (0, ry, 0)))
        leg.add('skin', tube((0, .1, .1), (0, 1.1, 1.4), .09, .06, 12)); leg.add('skin', ball((0, 1.1, 1.4), .09))
        kn = leg.child(Node('SpiderKnee%d' % i, (0, 1.1, 1.4))); kn.add('skin', tube((0, 0, 0), (0, -2.4, .9), .06, .018, 10))
    roots.append(sp)
    sa = Node('SpiderAbdomen', (0, 1.0, -.9)); sa.add('flesh', ellipsoid((0, 0, 0), (.5, .45, .8), 32, 20)); roots.append(sa)
    write_glb('models/monster.glb', roots)

# ================= ITEMS =================
def items():
    mat('silver', 0xb9bec6, .3, .9); mat('black', 0x15151a, .4, .2); mat('screen_blue', 0x10204a, .2, 0, (0x2a6fff, .9))
    mat('screen_green', 0x0a2a14, .2, 0, (0x33ff88, .9)); mat('paper', 0xe8e0c8, .9, 0, (0xe8e0c8, .12)); mat('ink', 0x222222, .9)
    mat('red', 0xc02020, .7, 0, (0xc02020, .2)); mat('cover', 0x2a4ab8, .7, 0, (0x2a4ab8, .18)); mat('bill', 0x3fae4f, .8, 0, (0x3fae4f, .2))
    mat('band', 0xc8a860, .9); mat('bucket', 0x3fc4d4, .4, .2, (0x3fc4d4, .2)); mat('water', 0x7fe8ff, .1, 0, (0x7fe8ff, .35))
    mat('pencil', 0xe8c030, .6); mat('strap', 0x25252c, .9); mat('lens', 0x0a0a14, .05, .5, (0x6a8aff, .25)); mat('keys', 0x2a2a30, .6)
    roots = []
    n = Node('ipad'); n.add('silver', box((0, 0, 0), (.62, .03, .44))); n.add('screen_blue', box((0, .0155, 0), (.56, .004, .38)))
    n.add('black', tube((0, .018, -.2), (0, .02, -.2), .008, .008, 8)); roots.append(n)
    n = Node('computer'); n.add('silver', box((0, .38, 0), (.7, .44, .05))); n.add('screen_blue', box((0, .38, .027), (.64, .36, .004)))
    n.add('silver', tube((0, .0, -.02), (0, .16, -.02), .035, .03, 10)); n.add('silver', ellipsoid((0, .0, -.02), (.16, .015, .12), 20, 8))
    n.add('keys', box((0, -.005, .3), (.52, .02, .17)))
    for r in range(4):
        for c in range(12): n.add('silver', box((-.235 + c * .043, .01, .245 + r * .037), (.034, .012, .03)))
    n.add('silver', ellipsoid((.4, 0, .3), (.035, .018, .055), 12, 8)); roots.append(n)
    n = Node('iphone'); n.add('black', box((0, 0, 0), (.22, .022, .44))); n.add('screen_blue', box((0, .0118, 0), (.2, .004, .41)))
    for k, (x, z) in enumerate(((-.05, -.17), (.05, -.17), (0, -.1))): n.add('silver', tube((x, -.011, z), (x, -.02, z), .028, .028, 14))
    roots.append(n)
    n = Node('vr'); n.add('black', box((0, 0, 0), (.52, .24, .22))); n.add('silver', box((0, 0, .112), (.46, .2, .01)))
    for sx in (-1, 1): n.add('lens', tube((sx * .11, 0, .12), (sx * .11, 0, .17), .075, .06, 18)); n.add('strap', tube((sx * .26, 0, -.05), (sx * .24, 0, -.1), .02, .02, 10))
    n.add('strap', pathtube([(-.25, 0, -.06), (-.28, 0, -.3), (-.1, 0, -.55), (.1, 0, -.55), (.28, 0, -.3), (.25, 0, -.06)], .028, 12))
    n.add('strap', pathtube([(0, .12, .0), (0, .22, -.25), (0, .1, -.55)], .025, 10)); roots.append(n)
    n = Node('chore'); n.add('paper', box((0, 0, 0), (.26, .008, .36))); n.add('red', box((0, .006, -.15), (.24, .004, .04)))
    for k in range(7):
        z = -.09 + k * .045; n.add('ink', box((.03, .006, z), (.15, .003, .008))); n.add('red', box((-.09, .006, z), (.025, .003, .025)))
    roots.append(n)
    n = Node('homework'); n.add('cover', box((0, 0, 0), (.28, .035, .36))); n.add('paper', box((.005, .004, .0), (.262, .026, .345)))
    n.add('cover', box((-.135, 0, 0), (.012, .04, .36))); n.add('pencil', tube((.05, .022, -.12), (.05, .022, .16), .008, .008, 8))
    n.add('ink', tube((.05, .022, .16), (.05, .022, .185), .008, .001, 8)); roots.append(n)
    n = Node('money')
    for k in range(7): n.add('bill', box((0, k * .0065, 0), (.19, .006, .09), Ry(.05 * math.sin(k * 2.3))))
    n.add('band', box((0, .022, 0), (.03, .05, .094))); roots.append(n)
    n = Node('bath'); n.add('bucket', lathe([(0, 0), (.1, 0), (.14, .3), (.15, .3), (.15, .31), (.13, .31), (.1, .0), (0, .0)][:6] + [(.13, .31), (.13, .27), (0, .27)], 32))
    n.add('water', lathe([(0, .268), (.128, .268), (.128, .272), (0, .272)], 28))
    n.add('silver', pathtube([(-.15, .3, 0), (-.12, .45, 0), (0, .5, 0), (.12, .45, 0), (.15, .3, 0)], .01, 8)); roots.append(n)
    n = Node('sonar'); n.add('black', box((0, 0, 0), (.13, .04, .22))); n.add('screen_green', tube((0, .02, .03), (0, .026, .03), .048, .048, 24))
    n.add('silver', tube((.05, .02, -.1), (.05, .2, -.12), .008, .005, 8))
    for k in range(3): n.add('red', tube((-.04 + k * .04, .02, -.07), (-.04 + k * .04, .028, -.07), .01, .01, 10))
    roots.append(n)
    write_glb('models/items.glb', roots)

# ================= ENVIRONMENT =================
def environment():
    mat('locker_metal', 0x4a5560, .45, .6); mat('locker_dark', 0x3a4550, .5, .6); mat('handle', 0x9a9a9a, .3, .9)
    mat('wood', 0x6a4a28, .85); mat('wood_dark', 0x5a3a1c, .8); mat('brass', 0xb08a3a, .3, .9)
    mat('crate_wood', 0x6b4f2a, .9); mat('crate_dark', 0x4a361c, .9); mat('barrel_metal', 0x3a4a4a, .5, .5); mat('hoop', 0x2a2a2a, .6, .6)
    mat('trim', 0xcfc79a, .8); mat('vent_m', 0x555555, .5, .7); mat('black', 0x15151a, .5); mat('cab', 0x6a7070, .5, .6); mat('fixture', 0x808080, .6, .4); mat('light_panel', 0x201a08, .5, 0, (0xffe9a0, .3))
    roots = []
    lk = Node('locker')
    lk.add('locker_metal', box((0, 2.17, 0), (.9, .06, .9))); lk.add('locker_metal', box((0, .03, 0), (.9, .06, .9)))
    lk.add('locker_metal', box((-.42, 1.1, 0), (.06, 2.2, .9))); lk.add('locker_metal', box((.42, 1.1, 0), (.06, 2.2, .9))); lk.add('locker_metal', box((0, 1.1, -.42), (.9, 2.2, .06)))
    lk.add('locker_dark', box((0, 1.5, -.3), (.78, .03, .26))); lk.add('handle', tube((-.3, 1.95, -.38), (.3, 1.95, -.38), .012, .012, 8))
    for x in (-.25, 0, .25): lk.add('handle', tube((x, 1.95, -.38), (x, 1.95, -.3), .01, .01, 8))
    dr = lk.child(Node('Door', (-.45, 1.1, .45)))
    for b in (box((.04, 0, 0), (.08, 2.1, .04)), box((.8, 0, 0), (.08, 2.1, .04)), box((.42, 1.03, 0), (.84, .08, .04)), box((.42, -1.03, 0), (.84, .08, .04)), box((.42, .02, 0), (.76, .06, .04)), box((.42, -.5, 0), (.76, .94, .03))): dr.add('locker_dark', b)
    for v in range(9): dr.add('locker_metal', box((.42, .12 + v * .105, .0), (.74, .06, .012), Rx(-.75)))
    for v in range(3): dr.add('locker_metal', box((.42, -.95 + v * .1, .0), (.74, .05, .012), Rx(-.75)))
    dr.add('handle', box((.74, 0, .045), (.04, .17, .04))); dr.add('handle', box((.42, .3, .03), (.16, .06, .01)))
    for k in range(3): dr.add('handle', tube((-.0, -.8 + k * .8, .0), (.0, -.7 + k * .8, .0), .02, .02, 10))
    roots.append(lk)
    tb = Node('table'); tb.add('wood', box((0, .86, 0), (1.8, .08, 1.2))); tb.add('wood_dark', box((0, .82, 0), (1.84, .03, 1.24)))
    for sx in (-1, 1):
        for sz in (-1, 1): tb.add('wood', tube((sx * .8, .82, sz * .5), (sx * .8, 0, sz * .5), .05, .03, 10))
    tb.add('wood_dark', box((0, .76, -.5), (1.6, .1, .03))); tb.add('wood_dark', box((0, .76, .5), (1.6, .1, .03)))
    tb.add('wood_dark', box((-.8, .76, 0), (.03, .1, 1.0))); tb.add('wood_dark', box((.8, .76, 0), (.03, .1, 1.0)))
    tb.add('wood_dark', box((0, .73, .53), (.6, .16, .04))); tb.add('brass', ellipsoid((0, .73, .56), (.03, .03, .02), 12, 8))
    tb.add('wood_dark', box((0, .73, .25), (.56, .13, .5))); tb.add('wood_dark', box((0, .35, 0), (1.55, .03, .03)))
    roots.append(tb)
    cr = Node('crate')
    for sx in (-.5, .5):
        for sz in (-.5, .5): cr.add('crate_dark', box((sx, .5, sz), (.1, 1, .1)))
    for y in (.17, .5, .83):
        for s in (-.5, .5): cr.add('crate_wood', box((0, y, s), (.8, .28, .04))); cr.add('crate_wood', box((s, y, 0), (.04, .28, .8)))
    cr.add('crate_dark', box((0, .02, 0), (.94, .04, .94))); cr.add('crate_dark', box((0, .98, 0), (.94, .04, .94)))
    cr.add('crate_dark', box((0, .5, .52), (1.05, .06, .03), Rz(.76))); roots.append(cr)
    br = Node('barrel'); R = lambda y: .44 + .06 * math.sin(math.pi * y)
    ys = np.linspace(0, 1, 14); br.add('barrel_metal', lathe([(0, 0)] + [(R(y), y) for y in ys] + [(0, 1)], 32))
    for y in (.12, .5, .88): r = R(y); br.add('hoop', lathe([(r - .01, y - .025), (r + .014, y - .025), (r + .014, y + .025), (r - .01, y + .025)], 32))
    roots.append(br)
    cl = Node('ceilinglight'); cl.add('fixture', box((0, -.025, 0), (1.0, .05, 1.0))); cl.add('light_panel', box((0, -.055, 0), (.9, .02, .9)))
    roots.append(cl)
    tr = Node('trim')
    for b in (box((0, .07, .0125), (1, .14, .025)), box((0, .15, .015), (1, .02, .03)), box((0, 3.12, .015), (1, .1, .03)), box((0, 3.05, .01), (1, .03, .02))): tr.add('trim', b)
    roots.append(tr)
    vt = Node('vent'); vt.add('vent_m', box((0, 2.6, .01), (.42, .28, .02)))
    for v in range(6): vt.add('hoop', box((0, 2.5 + v * .04, .022), (.36, .015, .01)))
    roots.append(vt)
    ot = Node('outlet'); ot.add('trim', box((0, .4, .006), (.08, .12, .012)))
    for dy in (-.03, .03): ot.add('hoop', box((0, .4 + dy, .013), (.03, .015, .004)))
    roots.append(ot)
    cb = Node('cabinet'); cb.add('cab', box((0, .5, 0), (.96, 1, .96)))
    for k in range(3): cb.add('locker_dark', box((0, .17 + k * .33, .485), (.88, .28, .02))); cb.add('handle', box((0, .22 + k * .33, .5), (.3, .03, .03)))
    roots.append(cb)
    ch = Node('chair'); ch.add('black', box((0, .5, 0), (.7, .1, .7))); ch.add('black', box((0, .85, -.32), (.7, .6, .08), Rx(-.15))); ch.add('handle', tube((0, .45, 0), (0, .12, 0), .04, .04, 10))
    for a in range(5):
        t = a * 2 * math.pi / 5; ch.add('handle', tube((0, .12, 0), (math.cos(t) * .38, .06, math.sin(t) * .38), .03, .025, 8)); ch.add('black', ball((math.cos(t) * .38, .04, math.sin(t) * .38), .04))
    roots.append(ch)
    write_glb('models/environment.glb', roots)

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for f in (monster, items, environment): MATS.clear(); f()
