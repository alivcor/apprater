########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

from logwriter import *
import sys

class bcolors:
    PINK = '\033[95m'
    LIME = '\033[0;32m'
    YELLOW = '\033[93m'
    VIOLET = '\033[0;35m'
    BROWN = '\033[0;33m'
    INDIGO = "\033[0;34m"
    BLUE = "\033[0;34m"
    LIGHTPURPLE = '\033[1;35m'
    LIGHTRED = '\033[1;31m'
    NORMAL = '\033[0;37m'
    SHARP = '\033[1;30m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    SLEEP = '\033[90m'
    UNDERLINE = '\033[4m'


    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.END = ''


def issueMessage(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.NORMAL}{.BOLD}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.NORMAL}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)


def issueSleep(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.SLEEP}{.BOLD}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.SLEEP}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)

def issueSharpAlert(text, logfilename, highlight=False):
    if highlight:
        toprint = "{.BOLD}(Iresium Engine)" + " : " + text + "{.END}"
        print toprint.format(bcolors, bcolors)
    else:
        toprint = "{.BOLD}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)


def issueError(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.FAIL}{.BOLD}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.FAIL}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)

def issueWelcome(logfilename):
    print "{.BLUE}{.BOLD} IRESIUM ENGINE {.END}".format(bcolors, bcolors, bcolors)

    with open(logfilename, 'a') as f:
        f.write("IRESIUM Engine")

    print "\n\n"
    with open(logfilename, 'a') as f:
        f.write("\n\n(Iresium Engine) : Welcome to Iresium v0.1")
    toprint = "{.BLUE}{.BOLD}(Iresium Engine){.END}" + " : " + "Welcome to Iresium v0.1"
    print toprint.format(bcolors, bcolors, bcolors)

def issueSuccess(text, logfilename, ifBold=False, highlight=False):
    if highlight:
        toprint = "{.LIME}{.BOLD}(Iresium Engine)" + " : " + text + "{.END}"
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        if ifBold:
            toprint = "{.LIME}{.BOLD}(Iresium Engine){.END}" + " : " + text
            print toprint.format(bcolors, bcolors, bcolors)
        else:
            toprint = "{.LIME}(Iresium Engine){.END}" + " : " + text
            print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)

def genLogFile(logfilename, ts, strts):
    toprint = "{.LIME}{.BOLD}(Iresium Engine){.END}" + " : " + "Logging all events to " + str(ts)
    print toprint.format(bcolors, bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + "Log File Created at : " + str(strts))
        f.write("\n(Iresium Engine) : " + "Logging all events to " + str(ts))


def issueWarning(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.BROWN}{.BOLD}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.BROWN}(Iresium Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(Iresium Engine) : " + text)

# issueWelcome()
# issueMessage("I'm glad you're here !")
# issueSleep("I'm turning to sleep mode.")
# issueSharpAlert("I'm back up. You need to look into this.")
# issueWarning("There's some problem with my core")
# issueError("I have to shutdown. Please mail the administrator the log file I've generated.")
#

def issueExit(logfilename, ts):
    toprint = "{.LIGHTPURPLE}{.BOLD}(Iresium Engine){.END}" + " : Shutting down the engine. Logs have been saved. Have a good day !"
    print toprint.format(bcolors, bcolors, bcolors)
    genpdfcmd = "python logwriter.py " + logfilename + " -S \"LOG FILE\" -A \"Iresium Engine\" -o logs/Iresium_Log_" + str(ts) + ".pdf"
    os.system(genpdfcmd)