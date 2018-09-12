import main
import sys

'''
Explorer routine

'''

# write your own if statement to call expl() with the arguments you like
if sys.argv[1] != 1:
    print 'This script requires a single string input that specifies the user of ctc.main.expl()'

main.expl( \
          strguser=sys.argv[1], \
         )
