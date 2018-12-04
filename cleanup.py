'''cleans up corrupted mesh and solution files and makes them ready for recomputations'''

import scipy.io as sio
import zlib
import os
import time
import socket

if socket.gethostname() == 'workstation1-room0436':
    path = '/home/constantin/cluster'
else:
    path = '/home/constantin'

path += '/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/' \
       'margins=0.003_0.003_0.003_0.003/N~logn/mu=7.8/sigma=0.2/x~GP/cov=squaredExponential/' \
       'l=0.08/sig_scale=1.2/r~lognGP/mu=-5.23/sigma=0.3' \
        + '/sigmaGP_r=0.4/l=0.05'

print('path == ', path)
bc = '/p_bc=0.0/u_x=1.0-0.0x[1]_u_y=1.0-0.0x[0]'

removing = True
mode = 'mesh'   # 'solution' or 'mesh'

N_max = 10000

for n in range(N_max + 1):
    try:
        if mode == 'mesh':
            filename = path + '/mesh' + str(n) + '.mat'
            tmp = sio.loadmat(filename)
            print('mesh ', str(n), ' fine.')
        elif mode == 'solution':
            filename = path + bc + '/solution' + str(n) + '.mat'
            tmp = sio.loadmat(filename)
            print('solution ', str(n), ' fine.')
        elif mode == 'microstructure':
            filename = path + '/microstructureInformation' + str(n) + '.mat'
            tmp = sio.loadmat(filename)
            print('microstructure ', str(n), ' fine.')
        else:
            print('Unknown mode!')
    except (FileNotFoundError, sio.matlab.miobase.MatReadError, OSError):
        if mode == 'mesh':
            print('mesh ', str(n), 'not found or still empty.')
        elif mode == 'solution':
            print('solution ', str(n), 'not found or still empty.')
        elif mode == 'microstructure':
            print('microstructure ' + str(n) + ' not found')
        else:
            print('Unknown mode!')
    except (TypeError, ValueError, zlib.error):
        if mode == 'mesh':
            print('mesh ', str(n), 'corrupted!')
        elif mode == 'solution':
            print('solution ', str(n), 'corrupted!')
        elif mode == 'microstructure':
            filename = path + bc + '/microstructureInformation' + str(n) + '.mat'
            tmp = sio.loadmat(filename)
            print('microstructure ', str(n), ' corrupted!')
        if removing:
            print('Remove corrupted file...')
            os.remove(filename)
            print('... file removed.')
            if mode == 'mesh':
                # for 'solution', this should be done manually by cleaning/removing 'computation_started/txt'
                print('Moving microstructureInformation' + str(n) +
                      '.mat to microstructureInformation_nomesh' + str(n) + '.mat ...')
                os.rename(path + '/microstructureInformation' + str(n) + '.mat',
                          path + '/microstructureInformation_nomesh' + str(n) + '.mat')
                print('... file moved.')



            # time.sleep(1)



