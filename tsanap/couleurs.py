class col:
    noir=black=[0,0,0]
    blanc=white=[255,255,255]
    rouge=red=[255,0,0]
    vert=green=[0,255,0]
    bleu=blue=[0,0,255]
    cyan=[0,255,255]
    magenta=[255,0,255]
    jaune=yellow=[255,255,0]
    def new(hexadecimal='000000',tipe='rgb'):
        '''
            Couleur héxadécimale en RGB par défaut.
            Couleur héxadécimale en BGR si c'est spécifié sur le type.
            ---
            Retourne une couleur en RGB
        '''
        if type(hexadecimal) == int:
            hexadecimal = f'{hexadecimal:x}'
        hexadecimal= hexadecimal.replace('#','')
        if tipe.lower()=='bgr':
            b, g, r = int(hexadecimal[0:2],base=16), int(hexadecimal[2:4],base=16), int(hexadecimal[4:6],base=16)
        else:
            r, g, b = int(hexadecimal[0:2],base=16), int(hexadecimal[2:4],base=16), int(hexadecimal[4:6],base=16)
        return[r,g,b]