import os
import re

def conv():
    cpt = 0
    for dirpath, dirnames, filenames in os.walk('.'):
        for dirname in dirnames:
            if re.match(r'^[A-Z][a-z]*_[A-Z][a-z]*$', dirname) is None and ' ' in dirname:
                print(f"Le nom du dossier '{dirname}' est mal formaté.")
                cpt +=1
    if cpt <= 1:        
        print(f"Il reste {cpt} dossier à formater")
    else:
        print(f"Il reste {cpt} dossiers à formater")
    return cpt