import SEAM
import SAM
import SYNAN
import DISAM
import SYNERR
import sys, getopt
import pickle, math, string
import numpy as np

def rate_description(app_description):
    try:
        printable = set(string.printable)
        cleaned_description = filter(lambda x: x in printable,app_description)
        with open("tempdesc.txt",'w') as f:
            f.write(cleaned_description)
        EssayFileName = "tempdesc.txt"
        DataFileName = "appdescriptions.csv"
        seam_score = SEAM.performLSA(EssayFileName, DataFileName)
        sam_score = SAM.performSA(EssayFileName, DataFileName)
        synan_score = SYNAN.scoreSYN(EssayFileName, DataFileName)
        disam_score = DISAM.scoreDiscourse(EssayFileName, DataFileName)
        synerr_score = SYNERR.scoreSYNERR(EssayFileName)
        # print seam_score
        # print sam_score
        # print synan_score
        # print disam_score
        # print synerr_score
        if math.isnan(disam_score):
            disam_score = 6

        calibrator = pickle.load(open("calibrated_model.sav", 'rb'))
        # print "FINAL SCORE : ", int(0.19738706*seam_score + 0.12756882*sam_score + 0.465254231*synan_score + 0.03680639*disam_score + 0.0728261*synerr_score)
        scoref = int(
            calibrator.predict(np.array([seam_score, sam_score, synan_score, disam_score, synerr_score]).reshape(1, -1)))
        # with open("/Users/abhinandandubey/Documents/untrase.txt", 'w') as fui:
        #     fui.write(str(scoref))
    except:
        scoref = 6
    return scoref
