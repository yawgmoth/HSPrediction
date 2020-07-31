import sys
import hsreplay.document
import os
import io
import random
import shutil

def analyzeone(fname,i):
    print(i, fname)
    xmlf = io.open(fname, "r", encoding="utf-8")
    doc = hsreplay.document.HSReplayDocument.from_xml_file(xmlf)
    t = doc.to_packet_tree()[0]
    xmlf.close()
    
def analyzeall(dname):
    i = 0
    files = os.listdir(dname)
    n = len(files)
    random.shuffle(files)
    training = int(round(n*0.62))
    test = int(round(n*0.19))
    validation = int(round(n*0.19))
    print(training, test,validation)
    trset = files[:training]
    teset = files[training:training+test]
    valset = files[training+test:]
    print(trset[0], " to ", trset[-1])
    print(teset[0], " to ", teset[-1])
    print(valset[0], " to ", valset[-1])
    
    os.makedirs("training")
    os.makedirs("test")
    os.makedirs("validation")
    for f in trset:
        shutil.copy2(os.path.join(dname, f), os.path.join("training", f))
    for f in teset:
        shutil.copy2(os.path.join(dname, f), os.path.join("test", f))
        
    for f in valset:
        shutil.copy2(os.path.join(dname, f), os.path.join("validation", f))
    
    
if __name__ == "__main__":
   analyzeall(sys.argv[1])