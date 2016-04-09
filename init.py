from pydub import AudioSegment
from scikits.talkbox.features import mfcc
import os, glob, scipy.io.wavfile, random
import numpy as np
import matplotlib.pyplot as plt

GENRES = ['soft', 'metal', 'pop']
FORMATS = ['mp3']
BASE_DIR = '/Users/xiaobaby/Desktop/shuffler'

#convert all music to wav
def to_wav():
    for genre in GENRES:
        for format in FORMATS:
            genre_dir = os.path.join(BASE_DIR, genre, '*.' + format)
            file_list =  glob.glob(genre_dir)
            wav_list = glob.glob(os.path.join(BASE_DIR, genre, '*.' + 'wav'))
            
            for fn in file_list:
                if fn[:-4] + '.wav' not in wav_list:
                    #place if conditions to convert formats other than mp3
                    sound = AudioSegment.from_mp3(fn)
                    sound.export(fn[:-4] + '.wav', format = 'wav')
                    print 'created ', fn[:-4] + '.wav'
        
        
#convert wav to .ceps  call create_ceps()
def write_ceps(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Written to %s" % data_fn)
def create_ceps():
    for genre in GENRES:
        wav_list = glob.glob(os.path.join(BASE_DIR, genre, '*.' + 'wav'))
        npy_list = glob.glob(os.path.join(BASE_DIR, genre, '*.' + 'npy'))
        npy_list = [name[:-9] for name in npy_list]
        for fn in wav_list:
            if fn[:-4] not in npy_list:
                sample_rate, X = scipy.io.wavfile.read(fn)
                ceps, mspec, spec = mfcc(X)
                write_ceps(ceps, fn)
                print 'created ', fn[:-4] + '.ceps.npy'

#feature extraction
def read_ceps(genre_list = GENRES, base_dir=BASE_DIR):
    X, y = [], []
    names = [] #song names
    for genre in genre_list:
        for fn in glob.glob(os.path.join(
                            base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(np.mean(
                     ceps[int(num_ceps*0.1):int(num_ceps*0.9)], axis=0))
            y.append(genre)
            names.append(os.path.basename(fn)[:-9])
    return np.array(X), np.array(y), names
    
from sklearn import svm
def train_svm(X, y, c = 1.0):
    clf = svm.SVC(C = c, probability=True)
    clf.fit(X, y) 
    return clf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
def train_randomforest(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    return rf
    
def roc(clf, testX, output):
    proba = clf.predict_proba(testX)
    fig, axis = plt.subplots(1,len(GENRES))
    fig.set_figheight(5)
    fig.set_figwidth(len(GENRES) * 5)
    fig.suptitle(clf.__class__.__name__)
    for i, genre in enumerate(GENRES):     
        proba_label = proba[:,clf.classes_ == genre]
        fpr, tpr, roc_thres = roc_curve(scipy.asarray(output==genre, dtype=int), proba_label)
        axis[i].plot(fpr, tpr, color = 'red')
        axis[i].set_ylim([0, 1.1])
        axis[i].set_title(genre + ' vs rest')
    fig.show()
    
def main():
    #convert mp3 to wav
    #to_wav()
    #convert wav to ceps.npy
    #create_ceps()
    
    X, y, names = read_ceps()
    X = list(X)
    y = list(y)

    testX, testY, testN = [], [], []
    #randomly select 25% as testing data (no replacement)
    for i in range(len(X)/4):
        n = random.randint(0, len(X) - 1)
        testX.append(X.pop(n))
        testY.append(y.pop(n))
        testN.append(names.pop(n))
    
    #svm
    clf = train_svm(X, y)
    output = clf.predict(testX)
    print 'actual ' , testY
    print 'result from svm' , output
    print [testY[i] == output[i] for i in range(len(output))]
    
    #random forest
    rf = train_randomforest(X, y)
    rf_output = rf.predict(testX)
    print 'result from rf' , rf_output
    print [testY[i] == rf_output[i] for i in range(len(rf_output))]
    print testN

    #plot roc curves
    roc(clf, testX, output)
            
if __name__ == '__main__':
    main()
    