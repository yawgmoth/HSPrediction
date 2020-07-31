import hslog.export
import hsreplay.document
import sys
import hearthstone.enums
from hearthstone.enums import GameTag, Zone, Step
import hearthstone.cardxml
import matplotlib.pyplot as plt
import os
import os.path
import pdb
import traceback
import json
import argparse
import io

tn = hearthstone.enums.TAG_NAMES

valid_tags = set(item.value for item in hearthstone.enums.GameTag)

SLOWLIMIT = 20

def format_tag_value(tag, value):
    if tag == GameTag.ZONE:
        return Zone(value)
    if tag in [GameTag.STEP, GameTag.NEXT_STEP]:
        return Step(value)
    return value

class MyExporter(hslog.export.BaseExporter):
    last_option = 0    
    optcnt = 0
    dts = []
    cnts = []
    opts = None
    entities = {}
    slowcards = {}
    normalcards = {}
    slowcount = 0
    def __init__(self, t):
        self.current_player = None
        self.plays = {}
        self.last_dt = None
        self.last_opt_cnt = None
        super().__init__(t)
    def handle_choices(self, choices):
        #print("choices", choices)
        pass
    def handle_show_entity(self, entity):
        if  entity.card_id is not None:
            #print("showed", db[entity.card_id], entity.entity)
            MyExporter.entities[entity.entity] = entity.card_id
            
    def handle_tag_change(self, tagchange):
        if tagchange.tag in valid_tags:
            of = ""
            if tagchange.entity in MyExporter.entities and False:
                import pdb
                pdb.set_trace()
                of = db[MyExporter.entities[tagchange.entity]]
            tag = GameTag(tagchange.tag)
            #print(tag, format_tag_value(tag, tagchange.value), of)
            if tag == GameTag.CURRENT_PLAYER and tagchange.value == 1:
                self.current_player = tagchange.entity
                if self.current_player not in self.plays:
                    self.plays[self.current_player] = []
                self.plays[self.current_player].append([])
                self.last_dt = None
                self.last_opt_cnt = None
            if tag == GameTag.JUST_PLAYED and tagchange.value == 1:
                ent = tagchange.entity 
                if ent in MyExporter.entities:
                    self.plays[self.current_player][-1].append((get_card(MyExporter.entities[ent]), self.last_opt_cnt, self.last_dt))
            
    def handle_full_entity(self, entity):
        if  entity.card_id is not None:
            #print("created", db[entity.card_id])
            MyExporter.entities[entity.entity] = entity.card_id
    def handle_options(self, options):
        valid = list(filter(lambda o: o.error is None or o.error == "-1", options.options))
        MyExporter.last_option = options.ts 
        
        MyExporter.optcnt = len(valid)
        MyExporter.opts = valid
        #print("choosing between", len(valid), "options")
        
    def handle_send_option(self, option):
        dt = option.ts - MyExporter.last_option
        #print("selected", MyExporter.optcnt, dt)
        which = MyExporter.normalcards
        ents = list(map(lambda o: o.entity, MyExporter.opts))
        if dt.total_seconds() > SLOWLIMIT:
            which = MyExporter.slowcards
            MyExporter.slowcount += 1
        for e in ents:
            if e in MyExporter.entities:
                if MyExporter.entities[e] not in which:
                    which[MyExporter.entities[e]] = 0
                which[MyExporter.entities[e]] += 1
        MyExporter.dts.append(dt.total_seconds())
        self.last_dt = dt.total_seconds()
        self.last_opt_cnt = MyExporter.optcnt
        MyExporter.cnts.append(MyExporter.optcnt)
        



def analyzeone(fname,maxturn=5):
    xmlf = io.open(fname, "r", encoding="utf-8")
    doc = hsreplay.document.HSReplayDocument.from_xml_file(xmlf)
    t = doc.to_packet_tree()[0]
    xmlf.close()
    MyExporter.entities = {}
    exp = MyExporter(t)
    r = exp.export()
    id = os.path.basename(fname)[:-len(".hsreplay.xml")]
    decks = deck_db[id]
    for (p,d) in [(2, decks[0]),(3, decks[1])]:
        line = "%s, %d, "%(id, p)
        linen = "%s, %d, "%(id, p)
        allp = []
        allpn = []
        hastime = False
        for i,pls in enumerate(exp.plays[p][:maxturn]):
            for pl in pls:
                (card,_,dt) = pl
                if len(allp) < 33:
                    allp.append(str(card))
                    allpn.append(str(card_id(card)))
                    allp.append(str(dt))
                    if dt is not None:
                        hastime = True
                    allpn.append(str(-1 if dt is None else dt))
                    allp.append(str(i+1))
                    allpn.append(str(i+1))
        while len(allp) < 33:
            allp.append("")
            allpn.append("-1")
        line += ",".join(allp)
        line += "," + str(d) + "," + archetype_db[int(d)]
        
        linen += ",".join(allpn)
        linen += "," + str(d) + "," + archetype_db[int(d)]
        if allpn[0] != "-1":
            print(line, file=outf)
            outf.flush()
        
            print(linen, file=outfn)
            outfn.flush()
        
        if hastime:
            print(linen, file=outfnt)
            outfnt.flush()
    
 
deck_db = {}
archetype_db = {}

def build_deck_db(dbdir="./"):
    files = os.listdir(dbdir)
    for f in files:
        if f.endswith(".json") and "list" in f:
            with open(os.path.join(dbdir, f)) as jsonf:
                info = json.load(jsonf)
                for game in info["data"]:
                    deck_db[game["id"]] = (game["player1_archetype"],game["player2_archetype"])
  
def build_archetype_db(dbf="archetypes.json"):
    with open(dbf) as jsonf:
        info = json.load(jsonf)
        for deck in info:
            archetype_db[deck["id"]] = deck["name"]
    
def main(dirname, maxturn=5):
    build_deck_db()
    build_archetype_db()

    if dirname.endswith(".xml"):
        analyzeone(dirname, maxturn)
    else:
        files = os.listdir(dirname)
        for f in files:
            if f.endswith(".xml") and "CardDefs" not in f:
                print("analyzing", f)
                try:
                    analyzeone(os.path.join(dirname, f), maxturn)
                except Exception:
                    traceback.print_exc()
    
    for c in MyExporter.slowcards:
        scnt = MyExporter.slowcards[c]
        ocnt = MyExporter.normalcards.get(c, 0)
        total = scnt + ocnt
        print("%15s: %3d of %5d (%5.2f) - %s"%(c, scnt, total, scnt*1.0/total, db[c]))
    print("total slow events", MyExporter.slowcount)
    
    plt.scatter(MyExporter.cnts, MyExporter.dts)
    plt.title('Thinking time vs. available options')
    plt.xlabel('Option Count')
    plt.ylabel('Time')
    plt.show()
    
(db,xml) = hearthstone.cardxml.load()

dbl = list(db.values())
def card_id(repr):
    return dbl.index(repr)

def get_card(card):
    if card in db:
        return db[card]
    print(card)
    print("card not found")
    import pdb
    pdb.set_trace()
    sys.exit(0)
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Hearthstone replays.')
    parser.add_argument('fname', metavar='N', type=str,
                    help='A replay file or directory with replay files')
    parser.add_argument('--max-turn', '-n', dest='maxturn', action='store', default=5, type=int,
                        help='How many turns to process (default: 5)')

    args = parser.parse_args()
    suffix = ""
    if args.maxturn != 5:
        suffix = "_%d"%(args.maxturn)
    outf = open("data%s_%s.csv"%(suffix,args.fname.strip("/\\.")), "w")
    print("id, pnr, p1, dt1, t1, p2, dt2, t2, p3, dt3, t3, p4, dt4, t4, p5, dt5, t5, p6, dt6, t6, p7, dt7, t7, p8, dt8, t8, p9, dt9, t9, p10, dt10, t10, p11, dt11, t11, archetype, archetypename", file=outf)

    outfn = open("datan%s_%s.csv"%(suffix,args.fname.strip("/\\.")), "w")
    print("id, pnr, p1, dt1, t1, p2, dt2, t2, p3, dt3, t3, p4, dt4, t4, p5, dt5, t5, p6, dt6, t6, p7, dt7, t7, p8, dt8, t8, p9, dt9, t9, p10, dt10, t10, p11, dt11, t11, archetype, archetypename", file=outfn)
    
    outfnt = open("datant%s_%s.csv"%(suffix,args.fname.strip("/\\.")), "w")
    print("id, pnr, p1, dt1, t1, p2, dt2, t2, p3, dt3, t3, p4, dt4, t4, p5, dt5, t5, p6, dt6, t6, p7, dt7, t7, p8, dt8, t8, p9, dt9, t9, p10, dt10, t10, p11, dt11, t11, archetype, archetypename", file=outfnt)
    main(args.fname, args.maxturn)
    
    outf.close()
    outfn.close()
    outfnt.close()