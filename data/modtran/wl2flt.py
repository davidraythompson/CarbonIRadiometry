#!/usr/bin/env python
# David R Thompson 

import os, sys, argparse
import spectral
import scipy as s
import scipy.signal as signal
from scipy.stats import norm

def main():

    I = None
    description = 'Convert wavelength file format to a MODTRAN .flt';
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',metavar='WLFILE', 
            help='input wavelength .txt file')
    parser.add_argument('outfile',metavar='FLTFILE', 
            help='output .flt file')
    args = parser.parse_args()

    wlfile = s.loadtxt(args.infile)
    i,wls,fwhms = wlfile.T
    if all(wls<100):
      wls = wls*1000.0
      fwhms = fwhms*1000.0
    sigmas = fwhms/2.355
    
    span = 2.5 * (fwhms[0]) # nm
    steps = 201

    with open(args.outfile,'w') as fout:
      fout.write('Nanometer data for AVIRIS-NG sensor, via '+args.infile+'\n')
      for i, (wl, fwhm, sigma) in enumerate(zip(wls, fwhms, sigmas)):
      
        ws = wl + s.linspace(-span, span, steps)
        vs = norm.pdf(ws,wl,sigma)
        vs = vs/vs[int(steps/2)]
        wns = 10000.0/(ws/1000.0)
      
        fout.write('CENTER:  %6.2f NM   FWHM:  %4.2f NM\n' % (wl, fwhm))  
      
        for w,v,wn in zip(ws,vs,wns):
          fout.write(' %9.4f %9.7f %9.2f\n' % (w,v,wn))  

if __name__ == '__main__':
    main()


