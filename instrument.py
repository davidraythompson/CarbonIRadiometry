# David R. Thompson
import json
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sbg_utils import load_chn, rule_07, blackbody, local_fit
from isofit.core.common import load_spectrum, resample_spectrum, load_wavelen, json_load_ascii

        
class Instrument():
    
        
    def __init__(self, config_file):
        
        config = json_load_ascii(config_file)
        self.config = config
        
        # Define basic instrument model
        self.wavelength_file = config['wavelength_file']
        self.wl, self.fwhm = load_wavelen(self.wavelength_file)
        self.del_wl = np.concatenate((np.diff(self.wl), np.diff(self.wl)[-1:]))  # nm

        # calculate the a_omega value
        self.detector_pitch  = config['detector']['pitch_cm']
        self.f_number        = config['f_number']
        self.t_intp          = config['integration_seconds']
        
        # temperatures
        self.spectrometer_temp  = config['spectrometer']['temperature']              
        self.telescope_temp     = config['telescope']['temperature']     
        self.fpa_temp           = config['detector']['temperature']  
            
        # Additional detector parameters
        self.cutoff            = config['detector']['cutoff_microns']
        self.detector_rows     = config['detector']['rows']
        self.detector_columns  = config['detector']['columns']
        self.detector_wellsize = config['detector']['wellsize']
        self.rule7_derate      = config['detector']['derate']
        
        # Additional electronics parameters
        self.electrons_per_dn     = config['electronics']['electrons_per_dn']
        self.compression_error_dn = config['electronics']['compression_error_dn']
        self.read_noise           = config['electronics']['read_noise']
        self.number_of_reads      = config['electronics']['number_of_reads']

        # Calculate telescope transmissions
        self.mirror_1 = \
            self.load_resample(config['telescope']["Mirror_1_Efficiency"])
        self.mirror_2 = \
            self.load_resample(config['telescope']["Mirror_2_Efficiency"])
        self.mirror_3 = \
            self.load_resample(config['telescope']["Mirror_3_Efficiency"])
        self.telescope_transm = self.mirror_1 * self.mirror_2 * \
            self.mirror_3
 
        # Elements of spectrometer transmission
        self.dyson_block = \
            self.load_resample(config["spectrometer"]["Dyson_Block_Efficiency"])
        if "Bandpass_Filter" in config["spectrometer"]:
            self.bandpass_filter = \
                self.load_resample(config["spectrometer"]["Bandpass_Filter"])
        self.grating = \
            self.load_resample(config["spectrometer"]["Grating_Efficiency"]) 
        if "grating_derate" in config["spectrometer"]:
            self.grating = self.grating - config["spectrometer"]["grating_derate"]
        if "throughput_derate" in config["telescope"]:
            self.telescope_transm = self.telescope_transm * config["telescope"]["throughput_derate"]
        if "dark_electrons" in config["detector"]:
            self.dark_electrons = config["detector"]["dark_electrons"]
        else:
            self.dark_electrons = None
            
        # Negligible diffraction loss for carbon-I
        self.slit_loss = 1
        
        # Handle oversized slits (larger than the detector pitch).  This translates
        # directly to throughput
        slit_width_cm       = config["spectrometer"]["slit_width_cm"]
        if slit_width_cm > self.detector_pitch:
            self.slit_oversize = slit_width_cm / self.detector_pitch

        # Total spectrometer transmission
        self.spectrometer_transm = self.grating * self.dyson_block * \
                                   self.slit_loss * self.bandpass_filter
        
        wl,q = np.loadtxt(config['detector']["qe_file"]).T
        self.qe = splev(self.wl,splrep(wl,q,s=0.001))
        self.qe[self.qe>1]=1
        
        self.throughput_degradation = 1
        self.noise_degradation = 0


    def load_resample(self, name):
        wav, t = load_spectrum(name)
        fwhm = np.ones(self.wl.shape) * self.del_wl
        return resample_spectrum(wav, t, self.wl, fwhm)


    # Calculate A Omega
    def a_omega(self):
        detector_area = self.detector_pitch**2
        return detector_area * np.pi / 4.0 / (self.f_number**2) * self.slit_oversize
        
        
    # Total throughput       
    def efficiency(self):
        return self.telescope_transm * \
               self.spectrometer_transm * \
               self.qe * \
               self.throughput_degradation
 

    # Total detected energy from target
    def scene_energy(self, rdn):       
        w_per_nm_cm2_sr = 0.000001 * rdn 
        total_W = w_per_nm_cm2_sr * self.a_omega() * self.del_wl
        return total_W 
        
        
    # Total detected photons from the target
    def scene_signal(self, rdn):       
        hc = 1.986e-16  # J nm
        E = hc / self.wl
        alpha_obs = 0.000001 * rdn / E
        alpha_int = alpha_obs * self.a_omega() * self.del_wl * self.t_intp
        return alpha_int * self.efficiency()

        
    # Calculate detected signal due to photons emitted from telescope    
    def telescope_signal(self):
        telescope_emissivity = (1 - self.telescope_transm)  # Kirchoff's law
        alpha_tel_int = blackbody(self.wl, self.telescope_temp) * \
                            telescope_emissivity * \
                            self.a_omega() * self.del_wl * self.t_intp
        return alpha_tel_int * self.spectrometer_transm * self.qe

        
    # Noise due to photon emission from telescope
    def telescope_noise(self):
        return np.sqrt(self.telescope_signal())
        
        
    # Calculate detected signal due to photons emitted from spectrometer
    # Note that this is not dispersed (i.e. just the max emission across
    # the spectral range).  We bookkeep a conservative value here.
    # We conservatively ignore quantum efficiency.
    # Note - Rob's version uses a_omega here instead of the integration
    # "hemisphere * detector_area"
    def spectrometer_signal(self):
        spectrometer_emissivity = 1.0
        emission = blackbody(self.wl, self.spectrometer_temp) * \
                            spectrometer_emissivity * \
                            self.del_wl * self.t_intp 
        hemisphere     = np.pi
        detector_area  = self.detector_pitch**2
        alpha_spec_int = emission * hemisphere * detector_area           
        return np.ones(self.wl.shape) * sum(alpha_spec_int * self.qe)

        
    # Noise due to photon emission from spectrometer
    def spectrometer_noise(self):
        return np.sqrt(self.spectrometer_signal())
    
        
    # MeCdTe dark current  
    # Be conservative, multiply by x10
    def dark_signal(self):
        if self.dark_electrons is None:
            j_dark          = rule_07(self.cutoff, self.fpa_temp)[0] * 10  # A cm-2
            q               = 1.602e-19  # Coulomb electron charge
            detector_area   = self.detector_pitch**2
            i_dark          = detector_area * j_dark / q  # electrons per second
            return i_dark * self.t_intp * self.rule7_derate
        else:
            return self.dark_electrons * self.t_intp

        
    # Poisson statistics for dark noise
    def dark_noise(self):  
        return np.sqrt(self.dark_signal())

        
    # Tally all photons reaching the detector.  
    # In rob's version, dark current is bookkept here.
    def total_signal(self, rdn): 
        return self.scene_signal(rdn)   + \
               self.telescope_signal()  + \
               self.spectrometer_signal()

        
    # Photon noise via counting statistics
    def shot_noise(self, rdn): 
        return np.sqrt(self.total_signal(rdn))

        
    # Quantization and truncation error
    def compression_noise(self):
        return self.compression_error_dn * self.electrons_per_dn
    
        
    # Quantization and truncation error
    def quantization_noise(self):
        return self.electrons_per_dn / np.sqrt(12)
    
        
    def total_noise(self, rdn): 
        noise_per_read = np.sqrt(self.shot_noise(rdn)**2 + \
                      self.read_noise**2 + \
                      self.dark_noise()**2 + \
                      self.compression_noise()**2 + \
                      self.quantization_noise()**2)
        return noise_per_read / np.sqrt(self.number_of_reads) + self.noise_degradation      
                    
    
    def noise_const(self): 
        noise_per_read = np.sqrt(self.read_noise**2 + \
                      self.dark_noise()**2 + \
                      self.compression_noise()**2 + \
                      self.quantization_noise()**2)
        return noise_per_read / np.sqrt(self.number_of_reads)      
    
        
    # Calculate SNR for a given scene
    def snr(self, rdn): 
        return self.scene_signal(rdn) / self.total_noise(rdn)

        
    # Translate SNR to a Noise-equivalent Delta Radiance
    def nedl(self, rdn): 
        return rdn / self.snr(rdn)
    
        
    # Calculate total divergence, noise equiv. band depth
    def divergence(self, rdnA, rdnB):
        return np.sqrt(sum(pow((rdnB - rdnA) / self.nedl(rdnA), 2)))


    def export_spectra(self, fname, rdn):
        row_names = ["f_number","t_intp",
            "read_noise","number_of_reads","rule7_derate",
            "cutoff","detector_rows","detector_columns",
            "detector_wellsize",
            "electrons_per_dn",
            "detector_pitch","spectrometer_temp",
            "telescope_temp","fpa_temp"]
        n_fields,n_wl = len(row_names),len(self.wl)
        row_values = [str(getattr(self, r)) for r in row_names]
        row_names.extend(["" for i in range(n_wl-n_fields)])
        row_values.extend(["" for i in range(n_wl-n_fields)])
        col_fields = [("specifications",row_names),("",row_values),
                      ("",["" for i in range(n_wl)]),
                      ("wl",self.wl), 
                      ("del_wl",self.del_wl),
                      ("rdn",rdn), 
                      ("fpa_qe",self.qe),
                      ("trn_mirror_1",self.mirror_1),
                      ("trn_mirror_2",self.mirror_2),
                      ("trn_mirror_3",self.mirror_3),
                      ("trn_slit",self.slit_loss),
                      ("trn_grating",self.grating), 
                      ("trn_dyson_block",self.dyson_block),
                      ("telescope_signal_e",self.telescope_signal()),
                      ("spectrometer_signal_e",self.spectrometer_signal()),
                      ("scene_signal_e",self.scene_signal(rdn)),
                      ("total_signal_e",self.total_signal(rdn)),
                      ("read_noise_e",np.ones(n_wl) * self.read_noise),
                      ("dark_noise_e",np.ones(n_wl) * self.dark_noise()),
                      ("compression_noise_e",np.ones(n_wl) * self.compression_noise()),
                      ("quantization_noise_e",np.ones(n_wl) * self.quantization_noise()),
                      ("shot_noise",self.shot_noise(rdn)),
                      ("total_noise",self.total_noise(rdn)),
                      ("snr",self.snr(rdn))]
        with open(fname,'w') as fout:
            fout.write(",".join([str(n) for n,val in col_fields])+"\n")
            for i,wl in enumerate(self.wl):
                fout.write(",".join([str(v[i]) for n,v in col_fields])+"\n")

