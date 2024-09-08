# Required packages:
# >>> opencv-python
# >>> multimethod
# >>> numpy
# >>> sty

from multimethod import multimethod
import numpy as np, cv2, random as rd, copy, time, os
from sty import Style, RgbFg, fg

try: from path_functs import *
except: from tsanap.path_functs import *

try: from terminal import *
except: from tsanap.terminal import *

try: from couleurs import *
except: from tsanap.couleurs import *

try: from calculs import *
except: from tsanap.calculs import *

try: from _vars_ import *
except: from tsanap._vars_ import *

try: import pip
except: import tsanap.pip

try: os.environ["XDG_SESSION_TYPE"] = "xcb"
except: pass

def debug(*args, **kwargs) -> None:
    '''Just a print function with another name.'''
    return print(*args, **kwargs)

def fusionImages(img, img_base, pos=[0, 0]):
    '''
    Prend:
    ------
    :img: ``np.array``\n
    :img_base: ``np.array``\n
    Renvoie:
    --------
    ``np.array``\nImage.
    '''
    pos = [round(v) for v in pos]
    x_offset, y_offset = pos
    try:
        img_base[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        return img_base
    except (IndexError, ValueError):
        sz_x, sz_y = len(img[0]), len(img)
        for x_ in range(sz_x):
            x = pos[0]+x_
            if x<0 or x>=len(img_base[0]): continue
            for y_ in range(sz_y):
                y = pos[1]+y_
                if y<0 or y>=len(img_base): continue
                img_base[y,x] = img[y_,x_]
        return img_base

class image:
    class boutton:
        def __init__(self, nom='-', coos=[[0, 0], []]) -> None:
            self.nom = nom
            self.coos = coos
            return
    def new_img(self=None, dimensions=screen, fond=[256 for _ in range(3)]) -> np.array:
        return np.full([round(v) for v in dimensions[::-1]]+[3], fond[::-1], np.uint8)
    def __init__(self, nom='image_python', img=None) -> None:
        self.nom = nom
        if type(img) == type(None):
            img = self.new_img()
        elif type(img) == image:
            img = img.img
        self.img = np.array(img)
        return
    def agrandis_img(self, cmb=2) -> None:
        '''
        Deprecated!
        -----------
        '''
        img = np.array([[[0,0,0] for x in range(len(self.img[0])*cmb)] for y in range(len(self.img)*cmb)])
        for y in range(len(img)):
            for x in range(len(img[0])):
                img[y,x] = self.img[y//cmb,x//cmb]
        self.img = img
        return
    def size(self, rev=False) -> [int, int]:
        return [len(self.img[0]), len(self.img)][::-1 if rev else 1] ## reverse True means [y,x] while False is [x,y]
    def __str__(self, ordre=True) -> str:
        img_str = ''
        n = 0
        cmb = len(self.img)*len(self.img[0])
        if ordre:
            mn_x, mx_x, st_x = 0, len(self.img), 1
            mn_y, mx_y, st_y = 0, len(self.img[0]), 1
        else:
            mn_x, mx_x, st_x = len(self.img)-1, -1, -1
            mn_y, mx_y, st_y = len(self.img[0])-1, -1, -1
        for x in range(mn_x, mx_x, st_x):
            for y in range(mn_y, mx_y, st_y):
                b, g, r = self.img[x, y]
                fg.custom = Style(RgbFg(r, g, b))
                img_str += f'{fg.custom}██'
                n += 1
            print(f'{len(self.img)}, {len(self.img[0])} - {n/cmb*100:4f}%', end='\r')
            img_str += '\n'
        img_str += fg.custom+fg.rs
        print(' '*20, end='\r')
        return img_str
    def montre(self, attente=0, destroy=False, fullscreen=False) -> int:
        '''
        In:
        ---
        :attente: ``int`` miliseconds\n
        :destroy: ``bool``\n
        :relocalisage_de_l_img: ``list | tuple `` of ``int`` [x, y]\n
        Out:
        ----
        ``int`` the waitkey (``ord(key)``)
        '''
        if fullscreen:
            cv2.namedWindow(self.nom, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.nom, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.nom, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.nom, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.nom, np.array(self.img, np.uint8))
        wk = cv2.waitKeyEx(attente)
        if destroy == True: cv2.destroyWindow(self.nom)
        return wk
    def close(self) -> None:
        return cv2.destroyWindow(self.nom)
    def copy(self): return image(img=copy.deepcopy(self.img))
    def visual_input(self, texte, ct, couleur=col.red, epaisseur=1, taille=1, police=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, lineType=0, fullscreen=False) -> str | None:
        tin = "" # Text input
        while True:
            img_s = image(nom=self.nom, img=copy.deepcopy(self.img))
            img_s.ecris(texte+tin, ct, couleur=couleur, epaisseur=epaisseur, taille=taille, police=police, lineType=lineType)
            wk = img_s.montre(1, False, fullscreen)
            match wk:
                case 27: return None
                case 8: tin=tin[:-1:]
                case 32: tin += ""
                case 13: break
                case -1: pass
                case _:
                    if wk<1000:
                        tin+=chr(wk)
        return tin
    def visual_input_option(self, texte, ct, opotions=["a", "b"], couleur=col.red, epaisseur=1, taille=1, police=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, lineType=0, fullscreen=False) -> str | None:
        tin = "" # Text input
        while True:
            img_s = image(nom=self.nom, img=copy.deepcopy(self.img))
            img_s.ecris(texte+tin, ct, couleur=couleur, epaisseur=epaisseur, taille=taille, police=police, lineType=lineType)
            wk = img_s.montre(1, False, fullscreen)
            match wk:
                case 27: return None
                case 8: tin=tin[:-1:]
                case 9: pass ## tab -> TODO 
                case 32: tin += ""
                case 13: break
                case -1: pass
                case _:
                    if wk<1000:
                        tin+=chr(wk)
        return tin
    def ferme(self) -> None:
        cv2.destroyWindow(self.nom)
    def imprime(self, ordre=True) -> None:
        print(self.__str__(ordre), end='')
        return
    def ligne(self, p1, p2, col=col.noir, ep=1, lineType=0) -> None:
        p1, p2 = [round(p) for p in p1], [round(p) for p in p2]; ep=round(ep)
        cv2.line(self.img, p1, p2, col[::-1], ep, [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3])
        return
    def rectangle(self, p1, p2, col=col.noir, ep=1, lineType=0) -> None:
        p1, p2 = [round(p) for p in p1], [round(p) for p in p2]; ep=round(ep)
        cv2.rectangle(self.img, p1, p2, col[::-1], ep if ep != 0 else -1, [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3])
        return
    def triangle(self, p1=ct_sg(p3, ct), p2=ct_sg(p4, ct), p3=ct_sg(ct, ch), couleur=col.noir, epaisseur=1, lineType=0):
        lineType = [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3]
        couleur = couleur[::-1]
        img = self.img
        p1 = [round(i) for i in p1]
        p2 = [round(i) for i in p2]
        p3 = [round(i) for i in p3]
        epaisseur = int(epaisseur)
        if epaisseur <= 0:
            if epaisseur == 0:
                epaisseur = 1
            else:
                epaisseur = abs(epaisseur)
            points = points_segment(p2, p3)
            for i in points:
                cv2.line(img, p1, i, couleur, epaisseur)
        cv2.line(img, p1, p2, couleur, epaisseur, lineType)
        cv2.line(img, p2, p3, couleur, epaisseur, lineType)
        cv2.line(img, p3, p1, couleur, epaisseur, lineType)
        self.img = img
        return
    def cercle(self, ct, rayon=10, col=col.noir, ep=1, lineType=0) -> None:
        ct = [round(p) for p in ct]; ep = round(ep)
        cv2.circle(self.img, ct, round(rayon), col[::-1], ep if ep != 0 else -1, [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3])
        return
    def ellipse(self, ct, rayons=[10, 10], col=col.noir, ep=1, lineType=0, anD=0, anF=360, ang=0) -> None:
        ct = [round(p) for p in ct]; ep = round(ep)
        cv2.ellipse(self.img, ct, [round(i) for i in rayons], ang, anD, anF, col[::-1], ep if ep != 0 else -1, [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3])
        return
    def arc(self, pa, pb, sagitta, col=col.noir, ep=1, lineType=0) -> None:
        if sagitta == 0: return self.ligne(pa, pb, col, ep, lineType)
        elif sagitta > 0:
            a = angleEntrePoints(pa, pb)
        else:
            a = angleEntrePoints(pa, pb) + 180; sagitta = abs(sagitta)
        ct = ct_sg(pa, pb)
        self.ellipse(ct, [dist(pa, pb)/2, sagitta], col, ep, lineType, 0, 180, a)
        return
    def arc2(self, pa, pb, sagitta, col=col.noir, ep=1, lineType=0) -> None: ## TODO : Fix imprecisions quand (|sagitta|<20 && sagitta != 0)
        s=sagitta;ct=ct_sg(pa,pb);d=dist(pa,pb)/2
        if abs(s)>d: return self.arc(pa,pb,s,col,ep,lineType)
        if s==0:return self.ligne(pa,pb,col,ep,lineType)
        elif s>0:a=angleEntrePoints(pa,pb)
        else:a=angleEntrePoints(pa,pb)+180;s=abs(s)
        x = ((d**2-s**2)/2)/s; r = x+s
        ctc = coosCercle(ct, x, a-90)
        a1,a2 = angleEntrePoints(ctc,pa), angleEntrePoints(ctc,pb)
        a_ = 360 if diff(a1,a2)>180 else 0
        return self.ellipse(ctc,(r,r), col, ep, lineType, min(a1,a2)+a_, max(a1,a2), 180)
    def parabole(self, a=1, b=0, c=0, puissance=2, couleur=col.bleu, epaisseur=10): ## TODO !!! ##
        p = puissance
        b = 0 - b
        pt = long/2, haut/2
        xct = pt[0]
        x = -xct - 10
        yct = pt[1]
        save = []
        while x <= xct:
            save.append([x + xct, yct - (a * (x ** p) + b * x + c)])
            if len(save) > 2:
                try:
                    self.ligne(save[-2], save[-1], couleur, epaisseur)
                except:
                    pass
            x += 1
        return
    def sauve_image(self, path='', nom_fichier=None) -> None:
        if nom_fichier == None: nom_fichier = self.nom
        if path != '':
            r = os.getcwd()
            os.chdir(path)
        print(nom_fichier)
        cv2.imwrite(nom_fichier, self.img)
        if path != '': os.chdir(r)
        return
    def ouvre_image(self, chemin) -> None:
        stream = open(f'{chemin}', "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return
    def set_img(self, img) -> None:
        self.img = np.array(img, np.uint8)
        return
    def ecris(self, texte, ct, couleur=col.red, epaisseur=1, taille=1, police=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, lineType=0) -> None:
        if True: ## Vars ##
            x1, y1 = ct
            x2, y2 = ct
            epaisseur = round(epaisseur)
            textes = list(enumerate(str(texte).split('\n')))
            valdef = cv2.getTextSize('Agd', police, taille, epaisseur)
            xxx = (x1+x2)/2
            yyy = (y1+y2)/2
            yyy -= round(valdef[0][1]*(len(textes)-1))
        for i, line in textes:
            tailles = cv2.getTextSize(line, police, taille, epaisseur)
            x = round(xxx-tailles[0][0]/2)
            y = round(yyy+tailles[1]/2)
            yy = y + i*tailles[0][1]*2
            cv2.putText(self.img, line, (x, yy), police, taille, couleur[::-1], epaisseur, [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA][lineType%3])
        return
    def is_closed(self) -> bool:
        '''Detect if the window is currently closed'''
        return cv2.getWindowProperty(self.nom,cv2.WND_PROP_VISIBLE)<1

class layout:
    class frame_:
        def __init__(self, img=image.new_img(fond=col.white), pos=[0,0], name='frame0') -> None:
            self.name = name
            self.img = image(img=copy.deepcopy(img))
            self.pos = pos
            return
        def __str__(self) -> str:
            return self.name
    def __init__(self, img=image.new_img(), frames=[], nom="Layout") -> None:
        self.nom = nom
        self.img = image(img=copy.deepcopy(img))
        self.frames = frames
        return
    def frame(self, img=image.new_img(fond=col.white, dimensions=[100, 100]), pos=[0,0], name=None):
        if name == None:
            name = 'frame' + str(len(self.frames))
        while name in self.frames: name += '_2'
        fenetre = self.frame_(img=img, pos=pos, name=name)
        self.frames.append(fenetre)
        return fenetre
    def montre(self, bords=False, frames=None, except_frames=[], fullscreen=True):
        img = image(self.nom, img=copy.deepcopy(self.img.img))
        if frames == None: frames = copy.deepcopy(self.frames)
        for frm in except_frames:
            ind = [i.name for i in frames].index(frm.name)
            if ind != -1: frames.pop(ind)
        cont = True
        for frame in frames:
            img.img = fusionImages(frame.img.img, img.img, frame.pos)
            if cont and type(bords) in [list, tuple]:
                for i in set(type(i) for i in bords):
                    if i not in [int, float]: cont=False
                if cont: img.rectangle(frame.pos, [frame.pos[0]+len(frame.img.img[0]), frame.pos[1]+len(frame.img.img)], bords, 3)
        return img.montre(1, fullscreen=fullscreen)
    def is_closed(self) -> bool:
        '''Detect if the layout is currently closed'''
        return cv2.getWindowProperty(self.nom,cv2.WND_PROP_VISIBLE)<1
    def size(self) -> [int, int]:
        return self.img.size()
    
def parabole_test():
    nf = "Paraboles"
    cv2.namedWindow(nf, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(nf, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    m = 10
    class prb:
        a=0
        b=0
        c=0
        p=0
    def get_a(v): prb.a=v
    def get_b(v): prb.b=v
    def get_c(v): prb.c=v
    def get_puissance(v): prb.p=v
    cv2.createTrackbar("a", nf, 0, m, get_a)
    cv2.setTrackbarMin("a", nf, -m)
    cv2.createTrackbar("b", nf, 0, m, get_b)
    cv2.setTrackbarMin("b", nf, -m)
    cv2.createTrackbar("c", nf, 0, m, get_c)
    cv2.setTrackbarMin("c", nf, -m)
    cv2.createTrackbar("puissance", nf, 0, m, get_puissance)
    t = 20; fs = True
    img = image(nf, img=image.new_img(fond=col.white))
    for x in range(0, 1920, t): img.ligne([x,0], [x,1080], col.noir, 1, 2)
    for y in range(0, 1080, t): img.ligne([0,y], [1920,y], col.noir, 1, 2)
    img_s = copy.deepcopy(img.img)
    while True:
        img.set_img(img_s)
        img.parabole(prb.a*t, prb.b*t, prb.c*t, prb.p, col.red, 10)
        wk = img.montre(attente=1, fullscreen=fs)
        match wk:
            case 27: break
            case 8: fs = not fs
        if img.is_closed(): break
def arc2_test():
    class s:
        v = 0
    def get_s(v): s.v = v
    p1 = [500, 500]
    p2 = [1000, 700]
    a,b=500,-300
    p3 = [p1[0]+a, p1[1]+b]
    p4 = [p2[0]+a, p2[1]+b]
    m=round(dist(p1, p2))
    nf = "Test"
    cv2.namedWindow(nf, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(nf, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("Sagitta", nf, 1, m, get_s)
    cv2.setTrackbarMin("Sagitta", nf, -m)
    fs = True
    while True:
        img = image(nf, img=demo(False))
        for p in [p1,p2,p3,p4]: img.cercle(p, 15, col.new("800080"), 0, 2)
        img.arc(p1, p2, s.v, col.green, 10, 2)
        img.arc2(p3, p4, s.v, col.green, 10, 2)
        wk = img.montre(fullscreen=fs, attente=1)
        if wk == 27: break
        elif wk == 8: fs=not fs
        if img.is_closed(): break
    return
def demo(exec=True):
    img = image(nom="Demo")
    pt = pt_sg([0, 0], screen)
    img.ellipse(pt, [500, 100], col.magenta, 10, 2)
    img.ligne([0, 0], screen, col.white, 10, 2)
    img.cercle([0, 0], 100, col.red, 0, 2)
    img.cercle(screen, 1000, col.red, 0, 2)
    fs = True
    while exec:
        wk = img.montre(fullscreen=fs, attente=1)
        match wk:
            case 27: break
            case 8: fs = not fs
            case 32: print(img.nom, end="\r")
        if img.is_closed(): break
    return img
if __name__ == '__main__':
    arc2_test()