import math, numpy as np
def n_entre(n, mi, ma) -> bool:
    '''
    Prend:
    ------
    :n: ``int`` | ``float``\n
    :mi: ``int`` | ``float``\n
    :ma: ``int`` | ``float``\n
    Renvoie:
    --------
    ``bool``: is n between mi and ma?
    '''
    return mi if n < mi else ma if n > ma else n
def decoupe(string):
    '''
    Prend:
    ------
    :nombre: ``complex`` ou ``str(complex)``\n
    Renvoie:
    --------
    ``float``
    '''
    out = ''
    for i in str(string):
        match i:
            case '('|')':pass
            case 'j':break
            case _:out+=i
    return float(out)
def ct_sg(pt1, pt2):
    '''
    Prend:
    ------
    :pt1: ``tuple (x, y)``\n
    :pt2: ``tuple (x, y)``\n
    Renvoie:
    --------
    ``ct``: ``tuple (x, y)``
    '''
    ct = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
    return ct
def ct_cr(p1, p2, p3, p4):
    '''
    # Centre_Carré
    Prend:
    ------
    :pt1: ``tuple (x, y)``\n
    :pt2: ``tuple (x, y)``\n
    :pt3: ``tuple (x, y)``\n
    :pt4: ``tuple (x, y)``\n
    Renvoie:
    --------
    ``ct``: ``tuple (x, y)``
    '''
    return ct_sg(ct_sg(p1, p2), ct_sg(p3, p4))
def pt_sg(p1, p2, m1=1, m2=1):
    '''
    Prend:
    ------
    :pt1: ``tuple (x, y)``\n
    :pt2: ``tuple (x, y)``\n
    :mult1: ``int``\n
    :mult2: ``int``\n
    Renvoie:
    --------
    ``ct``: ``tuple (x, y)``
    '''
    if m1 + m2 == 0: return(0, 0)
    pt = ((p1[n] * m1 + p2[n] * m2) / (m1+m2) for n in [0, 1])
    return [round(i) for i in pt]
def cts(pts):
    p1, p2, p3, p4 = pts
    return [ct_sg(p1, p2), ct_sg(p3, p4), ct_sg(p1, p3), ct_sg(p2, p4)]
def coosCercle(ct, rayon:int | float, angle, tipe="int"):
    '''
    Prend:
    ------
    :ct: ``tuple (x, y)``\n
    :rayon: ``int``\n
    :angle: ``int``\n\tInclinaison du cercle par rapport à son centre.\n
    Renvoie:
    --------
    ``tuple (x, y)``\n\tPosition du cercle à l\'angle précisé.
    '''
    angle = math.radians(angle)
    cos = float(decoupe(str(math.cos(angle))))
    sin = float(decoupe(str(math.sin(angle))))
    if tipe=='float':
        return [ct[0] + cos * rayon, ct[1] + sin * rayon]
    return [int(ct[0] + cos * rayon), int(ct[1] + sin * rayon)]
def coosEllipse(ct, rayons, angle):
    b, a = rayons
    p1, p2 = coosCercle(ct, min(a, b), angle), coosCercle(ct, max(a, b), angle)
    x, y = p1[0] - p2[0], p1[1] - p2[1]
    p3 = (p1[0] - x, p1[1])
    p4 = (p1[0], p1[1] - y)
    if a < b: return(p3)
    else: return(p4)
def dist(p1, p2):
    '''
    Calcule la distance entre p1 et p2
    '''
    a, b = p1
    c, d = p2
    diffX = abs(a - c)
    diffY = abs(b - d)
    dist = math.sqrt((diffX * diffX) + (diffY * diffY))
    dist = float(decoupe(dist))
    return dist
def angleEntrePoints(p1, p2):
    '''
    Calcule l'angle entre p1 et p2
    '''
    x1, y1 = p1
    x2, y2 = p2
    diffX = x1 - x2
    diffY = y1 - y2
    rotation = math.degrees(math.atan2(diffY, diffX))
    return(rotation)
def diff(n1, n2):
    '''
    Calcule la différence entre n1 et n2
    '''
    return abs(n1-n2)
def racine_carree(n):
    return float(decoupe(math.sqrt(n)))
def equation_2eme_degre(a, b, c):
    try:
        y1 = (-b + racine_carree(b**2 - 4*a*c)) / (2*a)
    except:
        y1 = 'r'
    try:
        y2 = (-b - racine_carree(b**2 - 4*a*c)) / (2*a)
    except:
        y2 = 'r'
    if y1 == 'r' and y2 == 'r':
        return(None)
    elif y1 == 'r':
        return(y2)
    elif y2 == 'r':
        return(y1)
    else:
        return y1, y2
def oppose(n):
    '''
    Donne la valeur opposée de n
    '''
    return(0 - n)
def points_segment(p1, p2):
    '''
    Prend:
    ------
    :p1: ``tuple`` ou ``list`` (x, y)\n
    :p2: ``tuple``  ou ``list`` (x, y)\n
    Renvoie:
    --------
    ``list`` de ``tuple`` (x, y)
    '''
    xa, ya = p1
    xb, yb = p2
    if xa == xb:
        dif = ya - yb
        numbs = range(abs(dif))
        out = []
        if dif < 0:
            for i in numbs:
                out.append([xa, yb - i])
        else:
            for i in numbs:
                out.append([xa, yb + i])
    else:
        xc = np.linspace(round(xa), round(xb), round(max(abs(xa - xb), abs(ya - yb))))
        yc = (yb - ya) / (xb - xa) * (xc - xa) + ya
        yc = [round(c) for c in yc]
        pos = 0
        out = []
        for i in xc:
            out.append([int(xc[pos]), int(yc[pos])])
            pos += 1
    return out
def float_range(start, stop, number=1):
    if stop==start: return [start for _ in range(number)]
    else:
        d = diff(start, stop)
        s = d/number
        o = []
        if start>stop:
            for n in range(number)[::-1]:
                o.append(stop+abs(s*n))
        else:
            for n in range(number):
                o.append(start+abs(s*n))
        return o
def range2(start, stop, step):
    if step == 0 or start == stop: return([])
    out = [start]
    a = start
    while True:
        a += step
        if a >= stop and step > 0:
            return(out)
        elif a <= stop and step < 0:
            return(out)
        out.append(a)
def moyenne(elementA, elementB, mult_elementA=1, mult_elementB=1, return_type='float') -> float | int:
    '''
    Fait la moyennes entre les elements
    Prend:
    ------
    :elementA: ``float | int``
    :elementB: ``float | int``
    :mult_elementA: ``float | int``
    :mult_elementB: ``float | int``
    :return_type: ``str`` (``"float" | "int"``)\n
    Renvoie:
    --------
    ``float``
    '''
    total_elements = mult_elementA + mult_elementB
    moyenne = ((elementA * mult_elementA) + (elementB * mult_elementB)) / total_elements
    if return_type != 'float':
        moyenne = int(moyenne)
    return(moyenne)
def clicked_in(pos, boutton):
    a_l_interieur = pos[0] >= boutton[0][0] and pos[0] <= boutton[1][0] and pos[1] >= boutton[0][1] and pos[1] <= boutton[1][1]
    return(a_l_interieur)