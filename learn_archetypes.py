import numpy as np
import sklearn.preprocessing as prep
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import csv 
import random
import sys
import torch
import torch.nn as nn
import hearthstone.cardxml
import time
import math
import os
import confusion_matrix_pretty_print as ppt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


(db,xml) = hearthstone.cardxml.load()

class FakeCard:
    def __init__(self, name):
        self.name = name 
    def __eq__(self, other):
        return self.name == other.name
    def __hash__(self):
        return hash(self.name)

dbl = list(db.values())

dbl.append(FakeCard("Secret 0"))
dbl.append(FakeCard("Secret 1"))
dbl.append(FakeCard("Secret 2"))
dbl.append(FakeCard("Secret 3"))
dbl.append(FakeCard("Secret 4"))
dbl.append(FakeCard("Secret 5"))

secret_ids = {0: len(dbl)-5, 1: len(dbl)-4, 2: len(dbl)-3, 3: len(dbl)-2, 4: len(dbl)-1, 5: len(dbl)-1}


def card_id(repr):
    return dbl.index(repr)

def get_card(card):
    card = int(card)
    return dbl[card]

# Adapted from https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA")
else:
    device = torch.device("cpu")
    print("CPU")
    
def prepare(data,n,enc,clsenc,labels, fname):
    def mov(tens):
        return tens.float().to(device)
    def conv(dt):
        return mov(torch.from_numpy(dt))
    
    preparedf = fname + ".%d.prepared"%n
    tpreparedf = fname + "%dt.prepared"%n
    tlpreparedf = fname + "%dtlabels.prepared"%n
    tclspreparedf = fname + "%dtclass.prepared"%n
    if os.path.exists(preparedf) and os.path.exists(tpreparedf) and os.path.exists(tlpreparedf) and os.path.exists(tclspreparedf):
        result = torch.load(preparedf)
        resultt = torch.load(tpreparedf)
        tlabels = torch.load(tlpreparedf)
        tcls = torch.load(tclspreparedf)
        return mov(result), mov(resultt), tlabels.long().to(device),tcls
        

    meantime = np.mean(alltimes)
    vartime = np.var(alltimes)
    tlabels = []
    result = np.zeros((n,len(data),len(enc.categories[0]) + len(clsenc.categories[0])))
    resultt = []
    tcls = []
    for row,d in enumerate(data):
        cls = d[0][-1]
        clse = clsenc.transform([[int(cls)]])
        crds = []
        newrow = []
        skip = False
        for i in range(n):
            card = d[i][0]
            entry = []
            carde = enc.transform([[card]])
            entry.append(carde[0])
            entry.append(clse[0])
            result[i][row] = np.concatenate(entry)
            t = d[i][1]
            if t > 0:
                entry.append(np.array([(t - meantime)/vartime]))
                newrow.append(np.concatenate(entry))
            else:
                skip = True
        
        if newrow and not skip:
            resultt.append(newrow)
            tlabels.append(labels[row])
            tcls.append(deck_db[deck_nr_db[labels[row]]][0].split()[-1])
            
    result = torch.from_numpy(result).float().to(device)
    torch.save(result, preparedf)
    
    if resultt:
        t = torch.tensor(resultt)
        resultt = t.permute(1,0,2).float().to(device)
    else:
        resultt = torch.tensor(resultt).float().to(device)
    torch.save(resultt, tpreparedf)
    tlabels = torch.tensor(tlabels).to(device)
    torch.save(tlabels, tlpreparedf)
    torch.save(tcls, tclspreparedf)
    return result,resultt,tlabels, tcls
    
class CardModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, encoder, classencoder):
        super(CardModel, self).__init__()
        self.encoder = encoder
        self.classencoder = classencoder
        self.hidden_dim = hidden_dim
        self.input_size = input_size 
        self.output_size = output_size
        self.rnn = nn.RNNCell(input_size, hidden_dim)   
        self.fc = nn.Linear(hidden_dim, output_size)
        self.d = nn.Dropout(0.2)
        self.sm = nn.Softmax(1)
    
    def forward(self, x):
        batchsize = x.shape[1]
        turns = x.shape[0]
        hidden = torch.zeros(batchsize,self.hidden_dim).to(device)
        
        for i in range(turns):
            input = x[i]
            hidden = self.rnn(input, hidden)
            if i != turns - 1:
                hidden = self.d(hidden)
           
        out = hidden.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return self.sm(out), hidden


#random.seed(1)

allcards = set(["-1"])
deck_db = {}
deck_nr_db = {}
clss = []

MAXCARDS = 11

alltimes = []


def read_file(fname, maxturn=2, n=0):
    data = []
    labels = []
    cls = []
    numbers = []
    batches = {}
    batchnumbers = {}
    maxncards = 0
    which = None
    if n > 0:
        which = torch.randperm(20556)[:n].tolist()
    with open(fname) as csvf:
        reader = csv.reader(csvf)
        header = True
        idx = -1
        for l in reader:
            if header:
                header = False 
            else:
                idx += 1
                if which and idx not in which:
                    continue
                #if l[1].strip() == "2":
                #    print("skip")
                #    continue
                deckname = l[-1].strip()
                deckid = int(l[-2].strip())
                if deckid not in deck_db:
                    decknr = len(deck_db)
                    deck_db[deckid] = (deckname,decknr)
                    deck_nr_db[decknr] = deckid
                else:
                    (_,decknr) = deck_db[deckid]
                clsname = deckname.split()[-1]
                if clsname not in clss:
                    clss.append(clsname)
                clsid = clss.index(clsname)
                seq = []
                ncards = 0
                for i in range(MAXCARDS):
                    card = l[2+3*i].strip()
                    if int(card) >= 0:
                        cardobj = get_card(card)
                        if cardobj.secret:
                            card = str(secret_ids[cardobj.cost])
                    allcards.add(card)
                    delay = float(l[2+3*i+1].strip())
                    if delay > 0:
                        alltimes.append(delay)
                    turn = int(l[2+3*i+2].strip())
                    if turn < maxturn and turn > 0:
                        seq.append([card,float(delay),int(turn),int(clsid)])
                        ncards += 1
                    else:
                        break
                maxncards = max(ncards, maxncards)
                if ncards > 0:
                    if ncards not in batches:
                        batches[ncards] = []
                        batchnumbers[ncards] = []
                    batches[ncards].append(seq[:])
                    if ncards == 2 and len(seq) == 1:
                        import pdb
                        pdb.set_trace()
                    batchnumbers[ncards].append(decknr)
                for i in range(MAXCARDS-ncards):
                    seq.append([-1,-1,-1,clsid])
                if ncards > 0:
                    labels.append(deckid)
                    numbers.append(decknr)
                    data.append(seq)
                    cls.append(clsname)
    print("Have at most", maxncards)
    return data,cls,labels,numbers,batches,batchnumbers
    
def find_max(probs):
    max = -1
    maxat = None
    for p in probs:
        if probs[p] > max:
            max = probs[p]
            maxat = p
    return (max,maxat)

def classify_statically(data, decks, labels, alldecks):
    correct = 0
    incorrect = 0
    alldecks = list(alldecks)
    predictions = []
    confusion = np.zeros((len(alldecks), len(alldecks)))
    for i in range(len(data)):
        game = data[i]
        for p in game:
            if p[0] != "-1":
                if int(p[-1]) in decks:
                   deck = decks[int(p[-1])]
        confusion[alldecks.index(deck),alldecks.index(labels[i])] += 1
        predictions.append(deck)
        if deck != labels[i]:
            incorrect += 1
        else:
            correct += 1
    return correct,incorrect,confusion,predictions
    
def classify_statisticially(data, probs, cls, labels, deckdist, cardprobs, alldecks, static_decks, f=max, bayes=True):
    correct = 0
    incorrect = 0
    predictions = []
    alldecks = list(alldecks)
    confusion = np.zeros((len(alldecks), len(alldecks)))
    for i in range(len(data)):
        game = data[i]
        gameprobs = {}
        staticdeck = None
        for p in game:
            if p[0] != "-1":
                if p[0] in probs:
                    for d in probs[p[0]]:
                        if deck_db[d][0].split()[-1] == cls[i]:
                            if d not in gameprobs:
                                gameprobs[d] = []
                            if bayes:
                                gameprobs[d].append(probs[p[0]][d]/cardprobs[p[0]])
                            else:
                                gameprobs[d].append(probs[p[0]][d])
                if int(p[-1]) in static_decks:
                   staticdeck = static_decks[int(p[-1])]
                            
        for d in gameprobs:
            if bayes:
                gameprobs[d] = f(gameprobs[d])*deckdist[d]
            else:
                gameprobs[d] = f(gameprobs[d])
        
        (prob,deck) = find_max(gameprobs)
        if not deck:
            deck = staticdeck
        
        confusion[alldecks.index(deck),alldecks.index(labels[i])] += 1
        predictions.append(deck)
        if deck != labels[i]:
            incorrect += 1
        else:
            correct += 1
    return correct,incorrect,confusion,predictions
    
def classify_rnn(model, batches, batchclss, alldecks):
    correct = 0
    incorrect = 0
    predictions = {}
    alldecks = list(alldecks)
    confusion = np.zeros((len(alldecks),len(alldecks)))
    for n in batches:
        if batches[n].shape[0] == 0: continue
        output,hidden = model(batches[n])
        (m,prediction) = output.max(1)
        diff = prediction == batchclss[n]
        for i in range(len(prediction)):
            confusion[alldecks.index(deck_nr_db[int(prediction[i].item())]),alldecks.index(deck_nr_db[batchclss[n][i].item()])] += 1
        whichtrue = diff.nonzero()
        correct += whichtrue.shape[0]
        incorrect += (prediction.shape[0] - whichtrue.shape[0])
        predictions[n] = prediction
        
    return correct,incorrect,confusion,predictions
    
def classify_rnn_combined(model, modelt, batches, batchclss, clsnames, alldecks):
    correct = 0
    incorrect = 0
    predictions = {}
    alldecks = list(alldecks)
    confusion = np.zeros((len(alldecks),len(alldecks)))
    for n in batches:
        if batches[n].shape[0] == 0: continue
        prediction = []
        for i in range(batches[n].shape[1]):
            if clsnames[n][i] in ["Warrior", "Priest"]:
                output,hidden = modelt(batches[n][:,i,:].unsqueeze(1))
            else:
                reduced = batches[n][:,i,:-1]
                output,hidden = model(reduced.unsqueeze(1))
            (m,pred) = output.max(1)
            
            confusion[alldecks.index(deck_nr_db[int(pred.item())]),alldecks.index(deck_nr_db[batchclss[n][i].item()])] += 1
            if pred.item() == batchclss[n][i].item():
                correct += 1
            else:
                incorrect += 1
            prediction.append(pred)
        predictions[n] = prediction
        
    return correct,incorrect,confusion,predictions
        
def distribution(labels):
    counts = {}
    for l in labels:
        if l not in counts:
            counts[l] = 0
        counts[l] += 1
    items = list(counts.items())
    items.sort(key=lambda i: -i[1])
    dist = {}
    for i in items:
        dist[i[0]] = i[1]*100.0/len(labels)
        print(deck_db[i[0]], i[1], i[1]*100.0/len(labels))
    return dist
        
def mostcommon(labels):
    result = {}
    counts = {}
    for l in labels:
        if l not in counts:
            counts[l] = 0
        counts[l] += 1
    items = list(counts.items())
    items.sort(key=lambda i: -i[1])
    for i in items:
        (name,id) = deck_db[i[0]]
        cls = name.split()[-1]
        clsid = clss.index(cls)

        if clsid not in result:
            result[clsid] = i[0]
    print(result)
    return result
        
def train(model, batches, batchnumbers, n_epochs=100, lr=0.01, testset=None, testlabels=None, totaltest=1, decks=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    batchkeys = list(batches.keys())
    splits = 100
    limit = 0.81
    for epoch in range(1, n_epochs + 1):
        random.shuffle(batchkeys)
        for b in batchkeys:
            if batches[b].shape[0] == 0: continue
            slices = batches[b].split(splits,1)
            targetslices = batchnumbers[b].split(splits)
            idxs = torch.randperm(len(targetslices))
            for idx in idxs:
                
                optimizer.zero_grad()
                input_seq = slices[idx]
                output, hidden = model(input_seq)
                target = targetslices[idx]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            #if epoch%50 == 0:
            #    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            #    print("Loss: {:.4f}".format(loss.item()))
        (correct,incorrect,conf,_) = classify_rnn(model, testset, testlabels, decks)
        print_score("%d: "%(epoch), correct, incorrect, totaltest)
        if (correct*1.0/(correct+incorrect) >= limit): return model
        if epoch > 100:
            limit = 0.8
        if epoch > 500:
            limit = 0.79
        if epoch > 900:
            limit = 0.785
    return model
    
    
def print_score(header, correct, incorrect, n):
    accuracy = correct * 1.0/(correct + incorrect)
    error = 1-accuracy
    conf = 1.96*math.sqrt(error*(1-error)/n)
    print("%s: correct"%(header), correct, "incorrect", incorrect, "perc. correct", accuracy, "+/-", conf)
    return accuracy
    
def pretty_print(confusion, labels, title, fname):
    #return
    df_cm = pd.DataFrame(confusion, index=labels, columns=labels)
    df_cmn = df_cm.div(df_cm.sum(axis=1), axis=0).fillna(0)
    fig = plt.figure(figsize=(10,8))
    sn.set(font_scale=1) # for label size
    
    sn.heatmap(df_cmn.iloc[:15,:15], vmin=0, vmax=1, annot=True, annot_kws={"size": 12}, fmt=".2f")
    
    
    print("Mispredictions:")
    for i in range(len(df_cm)):
        realclasses = list(zip(labels,df_cm.iloc[i]))
        realclasses.sort(key=lambda c: -c[1])
        alternatives = list(map(lambda c: "%s (%.2f)"%c, filter(lambda c: c[1] > 0.001, realclasses[:5])))
        if alternatives:
            print("Predicted %s, most common actual: %s"%(labels[i], ",".join(alternatives)))
    print("\n\n")
    
    print("Mispredictions:")
    for i in range(len(df_cm)):
        realclasses = list(zip(labels,df_cm.iloc[:,i]))
        realclasses.sort(key=lambda c: -c[1])
        alternatives = list(map(lambda c: "%s (%.2f)"%c, filter(lambda c: c[1] > 0.001, realclasses[:5])))
        if alternatives:
            print("Has %s, most common predicted: %s"%(labels[i], ",".join(alternatives)))
    print("\n\n")
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.title(title)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.subplots_adjust(left=0.2, bottom=0.25)
    plt.savefig(fname)
    plt.close(fig)
    

#def classification_report(*args, **kwargs):
#    return

def main(dataf="datan.csv", testf="datan_test.csv", holdoutf="datan_validation.csv", holdout=True, static=True, statistical=True, rnn=True):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    data,cls,labels,numbers,batches,batchnumbers = read_file(dataf)
    datatest,clstest,labelstest,numberstest,batchestest,batchnumberstest = read_file(testf)
    dataholdout,clsholdout,labelsholdout,numbersholdout,batchesholdout,batchnumbersholdout = read_file(holdoutf)
    
    accuracies = []
    mat = np.array(data)
    labs = np.array(labels)
    dist = distribution(labs)
    mattest = np.array(datatest)
    labstest = np.array(labelstest)
    
    matholdout = np.array(dataholdout)
    labsholdout = np.array(labelsholdout)
    
    totaltrain = mat.shape[0]
    totaltest = mattest.shape[0]
    totalholdout = matholdout.shape[0]
    
    n = len(labels)
    cards = list(allcards)
    cards.sort()
    inputs = len(cards)
    enc = prep.OneHotEncoder(categories=[cards], sparse=False)
    cards = mat[:,:,0]
    #cards.sort()
    enc.fit(cards.reshape((-1,1)))
    clsids = list(range(len(clss)))
    clsenc = prep.OneHotEncoder(categories=[clsids], sparse=False)
    inputs += len(clsids)
    clsenc.fit(np.array(clsids).reshape((-1,1)))
    categories = list(deck_db.keys())
    categories.sort(key=lambda c: -dist.get(c,0))
    categories = np.array(categories)
    labels_enc = prep.label_binarize(labs, classes=categories)
    outputs = len(categories)
    category_names = list(map(lambda c: deck_db[c][0], categories))
    decks = mostcommon(labs)
    if static:
       
        
        print("Static prediction")
        (correct,incorrect,confusion,prediction) = classify_statically(mat, decks, labs, categories)
        print_score("Training set", correct, incorrect, totaltrain)
        print(classification_report(labs,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labs,prediction))
        pretty_print(confusion, category_names, "Training Set (Static)", "plots/static_training.png")
        print("\n\n")

        (correct,incorrect,confusion,prediction) = classify_statically(mattest, decks, labstest, categories)
        print_score("Test set", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Static)", "plots/static_test.png")
        print("\n\n")
        
        if holdout:
            (correct,incorrect,confusion,prediction) = classify_statically(matholdout, decks, labsholdout, categories)
            accuracies.append(("Static", print_score("Holdout set", correct, incorrect, totaltest)))
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Static)", "plots/static_holdout.png")
            print("\n\n")
        
    if statistical:
        print("\n\n\nCalculate probabilities")
        cardprobs = {}
        cardprobs_num = {}
        ncards = 0
        t0 = time.time()
        for i in range(n):
            plays = mat[i]
            
            deck = deck_db[labs[i]]
            for p in plays:
                card = get_card(p[0])
                if int(p[0]) == -1:
                    continue
                if card not in cardprobs:
                    cardprobs[card] = {}
                    cardprobs_num[p[0]] = {}
                    
                if deck not in cardprobs[card]:
                    cardprobs[card][deck] = 0
                    cardprobs_num[p[0]][labs[i]] = 0
                if "total" not in cardprobs[card]:
                    cardprobs[card]["total"] = 0
                    cardprobs_num[p[0]]["total"] = 0
                cardprobs[card][deck] += 1.0
                cardprobs[card]["total"] += 1.0
                ncards += 1.0
                
                cardprobs_num[p[0]][labs[i]] += 1.0
                cardprobs_num[p[0]]["total"] += 1.0
        #print(cardprobs)
        
        cardprobs_norm = {}
        allcardprobs = {}
        allcardproblist = []
        
        for c in cardprobs_num:
            cardprobs_norm[c] = {}
            allcardprobs[c] = cardprobs_num[c]["total"]/ncards
            allcardproblist.append((c, cardprobs_num[c]["total"]/ncards))
            for d in cardprobs_num[c]:
                if d != "total":
                    cardprobs_norm[c][d] = cardprobs_num[c][d]*1.0/cardprobs_num[c]["total"]
        allcardproblist.sort(key=lambda c: -c[1])
        print(len(allcardproblist))
        for cp in allcardproblist[:50]:
            print(get_card(cp[0]), cp[1])
        t1 = time.time()
        print("Needed", t1-t0)
        
     
        
       
        #return
        
        #distribution(labstest)
        

        def product(array):
            result = 1
            for a in array:
                result *= a 
            return result
            
        def harmonic(array):
            result = 0
            for a in array:
                result += 1.0/a 
            return len(array)/result
            
        def geometric(array):
            return pow(product(array), 1.0/len(array))
        
        print("Training Set")        
        (correct,incorrect,confusion,prediction) = classify_statisticially(mat, cardprobs_norm, cls, labs, dist, allcardprobs, categories, decks)

    
        print_score("Training set", correct, incorrect, totaltrain)
        print(classification_report(labs,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labs,prediction))
        pretty_print(confusion, category_names, "Training Set (Statistical)", "plots/statistical_training.png")
        print("\n\n")
    
        
        print("Test Set")    
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks)
        print_score("Test set (max)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: max)", "plots/statistical_test_max.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=product)
        print_score("Test set (mul)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical)", "plots/statistical_test_mul.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=sum)
        print_score("Test set (add)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: add)", "plots/statistical_test_add.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=lambda d: sum(d)*1.0/len(d))
        print_score("Test set (avg)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: avg)", "plots/statistical_test_avg.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=min)
        print_score("Test set (min)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: min)", "plots/statistical_test_min.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=harmonic)
        print_score("Test set (harmonic)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: harmonic)", "plots/statistical_test_harmonic.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=geometric)
        print_score("Test set (geometric)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: geometric)", "plots/statistical_test_geometric.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=random.choice)
        print_score("Test set (random)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: random)", "plots/statistical_test_random.png")
        print("\n\n")
        
        if holdout:
            print("Holdout Set")    
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks)
            print_score("Holdout set (max)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: max)", "plots/statistical_holdout_max.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=product)
            accuracies.append(("Statistical", print_score("Holdout set (mul)", correct, incorrect, totalholdout)))
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical)", "plots/statistical_holdout_mul.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=sum)
            print_score("Holdout set (add)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: add)", "plots/statistical_holdout_add.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=lambda d: sum(d)*1.0/len(d))
            print_score("Holdout set (avg)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: avg)", "plots/statistical_holdout_avg.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=min)
            print_score("Holdout set (min)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: min)", "plots/statistical_holdout_min.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=harmonic)
            print_score("Holdout set (harmonic)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: harmonic)", "plots/statistical_holdout_harmonic.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=geometric)
            print_score("Holdout set (geometric)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: geometric)", "plots/statistical_holdout_geometric.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=random.choice)
            print_score("Holdout set (random)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: random)", "plots/statistical_holdout_random.png")
            print("\n\n")
        print("\n\n")
        
        print("Non-Bayes")
        print("Training Set")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mat, cardprobs_norm, cls, labs, dist, allcardprobs, categories, decks, bayes=False)

        print_score("Training set", correct, incorrect, totaltrain)
        print(classification_report(labs,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labs,prediction))
        pretty_print(confusion, category_names, "Training Set (Statistical)", "plots/statistical_nb_training.png")
        print("\n\n")
        
        print("Test Set")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, bayes=False)
        print_score("Test set (max)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: max)", "plots/statistical_nb_test_max.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=product, bayes=False)
        print_score("Test set (mul)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical)", "plots/statistical_nb_test_mul.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=sum, bayes=False)
        print_score("Test set (add)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: add)", "plots/statistical_nb_test_add.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=lambda d: sum(d)*1.0/len(d), bayes=False)
        print_score("Test set (avg)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: avg)", "plots/statistical_nb_test_avg.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=min, bayes=False)
        print_score("Test set (min)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: min)", "plots/statistical_nb_test_min.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=harmonic, bayes=False)
        print_score("Test set (harmonic)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: harmonic)", "plots/statistical_nb_test_harmonic.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=geometric, bayes=False)
        print_score("Test set (geometric)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: geometric)", "plots/statistical_nb_test_geometric.png")
        print("\n\n")
        (correct,incorrect,confusion,prediction) = classify_statisticially(mattest, cardprobs_norm, clstest, labstest, dist, allcardprobs, categories, decks, f=random.choice, bayes=False)
        print_score("Test set (random)", correct, incorrect, totaltest)
        print(classification_report(labstest,prediction,labels=categories,target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,prediction))
        pretty_print(confusion, category_names, "Test Set (Statistical: random)", "plots/statistical_nb_test_random.png")
        print("\n\n")
        
        if holdout:
            print("Holdout Set")    
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, bayes=False)
            print_score("Holdout set (max)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: max)", "plots/statistical_nb_holdout_max.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=product, bayes=False)
            print_score("Holdout set (mul)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical)", "plots/statistical_nb_holdout_mul.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=sum, bayes=False)
            print_score("Holdout set (add)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: add)", "plots/statistical_nb_holdout_add.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=lambda d: sum(d)*1.0/len(d), bayes=False)
            print_score("Holdout set (avg)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: avg)", "plots/statistical_nb_holdout_avg.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=min, bayes=False)
            print_score("Holdout set (min)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: min)", "plots/statistical_nb_holdout_min.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=harmonic, bayes=False)
            print_score("Holdout set (harmonic)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: harmonic)", "plots/statistical_nb_holdout_harmonic.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=geometric, bayes=False)
            print_score("Holdout set (geometric)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: geometric)", "plots/statistical_nb_holdout_geometric.png")
            print("\n\n")
            (correct,incorrect,confusion,prediction) = classify_statisticially(matholdout, cardprobs_norm, clsholdout, labsholdout, dist, allcardprobs, categories, decks, f=random.choice, bayes=False)
            print_score("Holdout set (random)", correct, incorrect, totalholdout)
            print(classification_report(labsholdout,prediction,labels=categories,target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,prediction))
            pretty_print(confusion, category_names, "Holdout Set (Statistical: random)", "plots/statistical_nb_holdout_random.png")
            print("\n\n")
        print("\n\n")
        
    
    if rnn:
        #print(inputs, outputs)
        print("prepare training set")
        tbatchnumbers = {}
        tbatches = {}
        tclsnames = {}
        for n in batches:
            batches[n],tbatches[n],tbatchnumbers[n],tclsnames[n] = prepare(batches[n], n, enc, clsenc, batchnumbers[n], dataf)
            batchnumbers[n] = torch.tensor(batchnumbers[n]).to(device)        
        
        print("prepare test set")
        tbatchnumberstest = {}
        tbatchestest = {}
        tclsnamestest = {}
        totalttest = 0
        for n in batchestest:
            batchestest[n],tbatchestest[n],tbatchnumberstest[n],tclsnamestest[n] = prepare(batchestest[n], n, enc, clsenc, batchnumberstest[n], testf)
            batchnumberstest[n] = torch.tensor(batchnumberstest[n]).to(device)
            totalttest += tbatchestest[n].shape[0]
            
        print("prepare holdout set")
        tbatchnumbersholdout = {}
        tbatchesholdout = {}
        tclsnamesholdout = {}
        totaltholdout = 0
        for n in batchesholdout:
            batchesholdout[n],tbatchesholdout[n],tbatchnumbersholdout[n],tclsnamesholdout[n] = prepare(batchesholdout[n], n, enc, clsenc, batchnumbersholdout[n], holdoutf)
            batchnumbersholdout[n] = torch.tensor(batchnumbersholdout[n]).to(device)
            totaltholdout += tbatchesholdout[n].shape[0]
        
        batchnrs = list(batches.keys())
        batchnrstest = list(batches.keys())
        batchnrsholdout = list(batches.keys())
        
        def concat(d, nrs):
            result = []
            for n in nrs:
                if n in d:
                    result.extend(map(lambda t: t.item(), d[n]))
            return result
            
        catlist = list(categories)
        

        labs = concat(batchnumbers, batchnrs)
        labslist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labs))
        labstest = concat(batchnumberstest, batchnrstest)
        labstestlist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labstest))
        
        labsholdout = concat(batchnumbersholdout, batchnrsholdout)
        labsholdoutlist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labsholdout))
        
        labst = concat(tbatchnumbers, batchnrs)
        labstlist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labst))
        labsttest = concat(tbatchnumberstest, batchnrstest)
        labsttestlist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labsttest))
        
        labstholdout = concat(tbatchnumbersholdout, batchnrsholdout)
        labstholdoutlist = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), labstholdout))
        
        sortedcategories = []
        catnames = list(map(lambda i: deck_db[deck_nr_db[i]], range(len(catlist))))
        nh = 54
        ne = 1000
        lr = 0.01
         
        print("\n\nParameters: hidden state: %d, epochs: %d, learning rate: %f"%(nh, ne, lr))
        model = CardModel(inputs, outputs, nh, enc, clsenc)
        model.to(device)
        t0 = time.time()
        model = train(model, batches, batchnumbers,ne,lr, batchestest, batchnumberstest, totaltest, categories)
        t1 = time.time()
        print("Needed", t1-t0)
        (correct,incorrect,confusion,prediction) = classify_rnn(model, batches, batchnumbers, categories)
        print_score("Training set", correct, incorrect, totaltrain)
        pred = concat(prediction,batchnrs)
    
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labslist,pred1,labels=range(len(categories)), target_names=category_names))
        
        print("Kappa: ", cohen_kappa_score(labs,pred))
        pretty_print(confusion, category_names, "Training Set (RNN hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_training.png"%(nh,lr))
        print("\n\n")
        
        (correct,incorrect,confusion,prediction) = classify_rnn(model, batchestest, batchnumberstest, categories)
        print_score("Test set", correct, incorrect, totaltest)
        pred = concat(prediction,batchnrstest)
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labstestlist,pred1,labels=range(len(categories)),target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labstest,pred))
        pretty_print(confusion, category_names, "Test Set (RNN hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_test.png"%(nh,lr))
        
        if holdout:
            (correct,incorrect,confusion,prediction) = classify_rnn(model, batchesholdout, batchnumbersholdout, categories)
            accuracies.append(("RNN", print_score("Holdout set", correct, incorrect, totalholdout)))
            pred = concat(prediction,batchnrsholdout)
            pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
            print(classification_report(labsholdoutlist,pred1,labels=range(len(categories)),target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labsholdout,pred))
            pretty_print(confusion, category_names, "Holdout Set (RNN hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_holdout.png"%(nh,lr))
        
        modelt = CardModel(inputs + 1, outputs, nh, enc, clsenc)
        modelt.to(device)
        print("\nWith times!")
        t0 = time.time()
        modelt = train(modelt, tbatches, tbatchnumbers, ne,lr,tbatchestest, tbatchnumberstest, totalttest, categories)
        t1 = time.time()
        print("Needed", t1-t0)
        (correct,incorrect,confusion,prediction) = classify_rnn(modelt, tbatches, tbatchnumbers, categories)
        print_score("Training set", correct, incorrect, totaltrain)
        pred = concat(prediction,batchnrs)
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labstlist,pred1,labels=range(len(categories)),target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labst,pred))
        pretty_print(confusion, category_names, "Training Set (RNN + times hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_training_t.png"%(nh,lr))
        print("\n\n")
        
        (correct,incorrect,confusion,prediction) = classify_rnn(modelt, tbatchestest, tbatchnumberstest, categories)
        print_score("Test set", correct, incorrect, totalttest)
        pred = concat(prediction,batchnrstest)
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labsttestlist,pred1,labels=range(len(categories)),target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labsttest,pred))
        pretty_print(confusion, category_names, "Test Set (RNN + times hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_test_t.png"%(nh,lr))
        
        if holdout:
            (correct,incorrect,confusion,prediction) = classify_rnn(modelt, tbatchesholdout, tbatchnumbersholdout, categories)
            accuracies.append(("RNN+times", print_score("Holdout set", correct, incorrect, totaltholdout)))
            pred = concat(prediction,batchnrsholdout)
            pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
            print(classification_report(labstholdoutlist,pred1,labels=range(len(categories)),target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labstholdout,pred))
            pretty_print(confusion, category_names, "Holdout Set (RNN + times hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_%d_%.3f_holdout_t.png"%(nh,lr))
        
        print("\n\n")
        
        print("Combination of the two models")
        (correct,incorrect,confusion,prediction) = classify_rnn_combined(model, modelt, tbatches, tbatchnumbers, tclsnames, categories)
        print_score("Training set", correct, incorrect, totaltrain)
        pred = concat(prediction,batchnrs)
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labstlist,pred1,labels=range(len(categories)),target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labst,pred))
        pretty_print(confusion, category_names, "Training Set (RNN combined hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_combined_%d_%.3f_training_t.png"%(nh,lr))
        print("\n\n")
        
        (correct,incorrect,confusion,prediction) = classify_rnn_combined(model, modelt, tbatchestest, tbatchnumberstest, tclsnamestest, categories)
        print_score("Test set", correct, incorrect, totalttest)
        pred = concat(prediction,batchnrstest)
        pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
        print(classification_report(labsttestlist,pred1,labels=range(len(categories)),target_names=category_names))
        print("Kappa: ", cohen_kappa_score(labsttest,pred))
        pretty_print(confusion, category_names, "Test Set (RNN combined hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_combined_%d_%.3f_test_t.png"%(nh,lr))
        
        if holdout:
            (correct,incorrect,confusion,prediction) = classify_rnn_combined(model, modelt, tbatchesholdout, tbatchnumbersholdout, tclsnamesholdout, categories)
            accuracies.append(("Static", print_score("Holdout set", correct, incorrect, totaltholdout)))
            pred = concat(prediction,batchnrsholdout)
            pred1 = list(map(lambda c: catlist.index(deck_nr_db[int(c)]), pred))
            print(classification_report(labstholdoutlist,pred1,labels=range(len(categories)),target_names=category_names))
            print("Kappa: ", cohen_kappa_score(labstholdout,pred))
            pretty_print(confusion, category_names, "Holdout Set (RNN combined hidden size: %d, lr: %.3f)"%(nh,lr), "plots/rnn_combined_%d_%.3f_holdout.png"%(nh,lr))
    
    return accuracies
    # for 95% confidence interval:
    # error +/- 1.96 * sqrt( (error * (1 - error)) / n)


if __name__ == "__main__":
    allaccuracies = {}
    iters = 10
    for i in range(iters):
        if len(sys.argv) > 1:
            acc = main(sys.argv[1])
        else:
            acc = main()
        for k in acc:
            (name,val) = k
            if name not in allaccuracies:
                allaccuracies[name] = 0
            allaccuracies[name] += val
        
        print("Summary after", i, "iterations")
        for name in allaccuracies:
            print(name, allaccuracies[name]*1.0/(i+1))
