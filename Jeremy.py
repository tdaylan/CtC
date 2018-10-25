import main, models, testfunc
import time

# gdat
inst = main.gdatstrt()

# variables:
inst.numbepoc = 1
inst.numbruns = 4
inst.fractest = 0.2


now = time.time()
# explore --> data gets stored in the gdat instance
# globlrun = testfunc.explore(instL, models.globl)



loclrun = testfunc.run_through_puts(inst, models.singleinput)

after = time.time()

print('elapsed time: ', (after-now)/60)

