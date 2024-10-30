 # David R. Thompson
import re
import scipy as s
from scipy import optimize
    
# -------------------------------------------------------------------------- 
#                             MODTRAN 6.0.0 Support                            
# -------------------------------------------------------------------------- 

# Parse a MODTRAN output file with critical coefficient vectors.  
#       These are:
#         wl      - wavelength vector
#         sol_irr - solar irradiance
#         sphalb  - spherical sky albedo at surface
#         transm  - diffuse and direct irradiance along the 
#                      sun-ground-sensor path
#         transup - transmission along the ground-sensor path only 
#       We parse them one wavelength at a time.
def load_chn(infile, coszen):

    with open(infile) as f:
        sols, transms, sphalbs, wls, rhoatms, transups = [], [], [], [], [], []
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i < 5:
                continue
            toks = line.strip().split(' ')
            toks = re.findall(r"[\S]+", line.strip())
            wl, wid = float(toks[0]), float(toks[8])  # nm
            solar_irr = float(toks[18]) * 1e6 * \
                s.pi / wid / coszen  # uW/nm/sr/cm2
            rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
            rhoatm = rdnatm * s.pi / (solar_irr * coszen)
            sphalb = float(toks[23])
            transm = float(toks[22]) + float(toks[21])
            transup = float(toks[24])
            sols.append(solar_irr)
            transms.append(transm)
            sphalbs.append(sphalb)
            rhoatms.append(rhoatm)
            transups.append(rhoatm)
            wls.append(wl)
    params = [s.array(i) for i in
              [wls, sols, rhoatms, transms, sphalbs, transups]]
    return tuple(params)

# -------------------------------------------------------------------------- 
#                             Physical constants                            
# -------------------------------------------------------------------------- 

k          = 1.380648e-23    # Boltzmann constant
q          = 1.60217733e-19  # elementary charge, in Coulombs
c_1        = 1.88365e32/s.pi # first rad. constant, photon radiance
c_2        = 14387690        # second radiometric constant
c_1watt    = 3.7417749e16/s.pi # first radiometric constant for W/cm2/sr/nm
hc         = 1.986e-16 # J nm

def sinc2(x):
  return pow(s.sinc(x),2)

# blackbody emission in oh / (sec cm2 sr nm)
def blackbody(wvl, T):
  return c_1 / pow(wvl,4) / (s.exp(c_2 / wvl / T)-1.0)

def delbb(wvl, T, dT):
  return c_1 / pow(wvl,4) * (1.0 / (s.exp(c_2/wvl/(T+dT))-1.0) -\
                             1.0 / (s.exp(c_2/wvl/T)-1.0))

# -------------------------------------------------------------------------- 
#                             Mercad Properties                            
# -------------------------------------------------------------------------- 

# "Rule 07" dark current level, for cutoff wavelength lambda given in microns
# via Tennant et al., Journal of Electronic Materials, Vol. 37, No. 9, 2008
def rule_07(cutoff, Temp_FPA):
  J0, Pwr, C = 8367.00001853855, 0.544071281108481, -1.16239134096245
  scale, thresh = 0.200847413564122, 4.63513642316149
  if cutoff >= thresh: 
    lambda_e = cutoff
  else:
    lambda_e = cutoff / (1.0 - pow(scale/cutoff - scale/thresh, Pwr))
  J_07 = J0 * s.exp(C*(1.24 * q / k / lambda_e / Temp_FPA))
  dJ_07 = -1.24 * q * C / k / lambda_e / Temp_FPA **2 * J_07
  return J_07, dJ_07

# Quantum efficiency curve at given wavelengths & cutoff, all in microns
# It is typically 0.8, and modeled as a polynomial near the cutoff 
# via Beletic et al., "Teledyne Imaging Sensors...", Proc. SPIE, 2008
def HgCdTe_qe(wvl_um, cutoff):
  max_qe        = 0.8
  qe            = max_qe * s.ones(wvl_um.shape)
  hmax          = 0.1 * cutoff # qe varies within 10% of cutoff
  mask          = wvl_um > (cutoff - hmax)
  x             = s.array([cutoff-2*hmax, cutoff-hmax, cutoff, cutoff+hmax])
  y             = [max_qe, max_qe, 0.5*max_qe, 0.0]
  qe[mask]      = s.polyval(s.polyfit(x,y,3), wvl_um[mask]) 
  return qe

# -------------------------------------------------------------------------- 
#                           Least-Squares Fitting                            
# -------------------------------------------------------------------------- 

def local_fit(wl, x, lib, rng):

    x[s.logical_not(s.isfinite(x))] = 0;
    wl[s.logical_not(s.isfinite(wl))] = 0;
    lib[s.logical_not(s.isfinite(lib))] = 0;
    
    # subset to our range of interest
    i1 = s.argmin(abs(wl-rng[0]))
    i2 = s.argmin(abs(wl-rng[-1]))
    x, wlf, lib = x[i1:i2], wl[i1:i2], lib[i1:i2];
    
    # Continuum level
    ends = s.array([0,-1], dtype=s.int32)
    p = s.polyfit(wlf[ends], x[ends], 1);
    xctm = s.polyval(p, wlf)
    xctmr = x / xctm - 1.0;

    # Continuum of library
    p = s.polyfit(wlf[ends], lib[ends], 1);
    lctm = s.polyval(p, wlf)
    lctmr = lib / lctm - 1.0;
    
    # Fit a scaling term
    def err(scale):
        return sum(pow(scale * lctmr - xctmr, 2));
                       
    scale = optimize.minimize_scalar(err, bracket=[0,1])
    libfit = (1.0 + scale.x * lctmr) * xctm;

    return wlf, libfit 
