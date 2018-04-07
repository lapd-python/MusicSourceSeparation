
import statistics
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def print_res():
    SARnew = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SAR.p')
    SARold = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SARo.p')
    SDRnew = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SDR.p')
    SDRold = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SDRo.p')
    SIRnew = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SIR.p')
    SIRold = np.load('../results/GRU_sskip_filt/inference_m3_i10plus/SIRo.p')
    # print('[%s]' % ', '.join(map(str, SARnew)))
    SARnew = [item for items in (item for items in SARnew for item in items) for item in items]
    SARold = [item for items in (item for items in SARold for item in items) for item in items]
    SDRnew = [item for items in (item for items in SDRnew for item in items) for item in items]
    SDRold = [item for items in (item for items in SDRold for item in items) for item in items]
    # SIRnew = [item for items in (item for items in SIRnew for item in items) for item in items]
    SIRold = [item for items in (item for items in SIRold for item in items) for item in items]
    SARnew = [x for x in SARnew if str(x) != 'nan']
    SARold = [x for x in SARold if str(x) != 'nan']
    SDRnew = [x for x in SDRnew if str(x) != 'nan']
    SDRold = [x for x in SDRold if str(x) != 'nan']
    # SIRnew = [x for x in SIRnew if ((str(x) != 'nan') & (str(x) != 'inf'))]
    SIRold = [x for x in SIRold if ((str(x) != 'nan') & (str(x) != 'inf'))]

    print(SIRnew)
    print('mean SAR old: ' + str(np.mean(SARold)))
    print('mean SAR new: ' + str(np.mean(SARnew)))
    print('mean SDR old: ' + str(np.mean(SDRold)))
    print('mean SDR new: ' + str(np.mean(SDRnew)))
    print('mean SIR old: ' + str(np.mean(SIRold)))
    print('mean SIR new: ' + str(np.mean(SIRnew)))
    print('median SAR old: ' + str(np.median(SARold)))
    print('median SAR new: ' + str(np.median(SARnew)))
    print('median SDR old: ' + str(np.median(SDRold)))
    print('median SDR new: ' + str(np.median(SDRnew)))
    print('median SIR old: ' + str(np.median(SIRold)))
    print('median SIR new: ' + str(np.median(SIRnew)))


    # for x in SDRnew:
    #     print((x[~np.isnan(x)]))
    #     plt.plot(x[0])
    #     plt.show()

if __name__ == '__main__':
    print_res()

# EOF