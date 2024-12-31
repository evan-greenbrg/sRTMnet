






import numpy as np
import argparse
import h5py
import subprocess
import os
import ray
import multiprocessing as mp


class resample():
    """ Resampling class where you can initialize and then use. It has a saftey logic against non finite
    elements in the data to interpoloate. 

    Args:
        wvs_a: High resolution data's spectral grid
        wvs_b: Instrument's resolution spectra grid
        fwhm_b: Instrument's FWHM
    """
    def __init__(self, wvs_a, wvs_b, fwhm_b):
        self.wvs_a = wvs_a
        self.wvs_b = wvs_b
        self.fwhm_b = fwhm_b
        self.get_transform_matrix()

    
    def get_transform_matrix(self):
        base_wl = np.array(self.wvs_a)
        self.base_wl = base_wl
        target_wl = np.array(self.wvs_b)
        self.target_wl = target_wl
        target_fwhm = np.array(self.fwhm_b)

        doTheResample = lambda id: self.spectrumResample(id, base_wl, target_wl, target_fwhm)
        ww = np.array([doTheResample(W) for W in range(len(target_wl))])
        self.transform_matrix = ww
    

    def process_data(self, ys, n_samples, num_processes=4):
        """
        Process data in parallel.
        Args:
            ys: The data to be processed.
            n_samples: The number of samples expected in ys.
            num_processes: The number of parallel processes to use.
        Returns:
            Processed data.
        """
        num_processes = mp.cpu_count()
        # Check dimensions and align correctly
        ys = self._reshape_data(ys, n_samples)

        # Initialize Ray
        #if not ray.is_initialized():
        #    ray.init(num_cpus=num_processes)

        # Dispatch tasks
        #result_ids = [self.process_single_sample.remote(self, ys[i, :]) for i in range(ys.shape[0])]
        #results = ray.get(result_ids)
        results = [self.process_single_sample(ys[i, :],i) for i in range(ys.shape[0])]

        return np.array(results)

    #@ray.remote(num_cpus=1)
    def process_single_sample(self, y, i):
        """
        Process a single sample. This method will be called in parallel.
        Args:
            y: A single sample from ys.
        Returns:
            The processed sample.
        """
        # Processing logic for a single sample
        # For example, this could be a call to self.__call__(y) or other processing
        if i % 100:
            print(i)
        return self.__call__(y)  # or other processing logic


    def _reshape_data(self, ys, n_samples):
        """
        Reshape data to the correct dimensions for convolution.
        """
        counter = 0
        max_attempts = 5
        while ys.shape[0] != n_samples:
            if counter >= max_attempts:
                raise ValueError("Unable to reshape data to the correct dimensions for convolution")
            ys = ys.T
            counter += 1
        return ys

    def __call__(self, y):
        # Convert input to 2D array and transpose if necessary

        # Convert input to 2D array
        spectrum = np.atleast_2d(y)

        # Check if transpose is necessary
        transpose_needed = spectrum.shape[0] == 1

        # Initialize an output array
        resampled_spectrum = np.zeros(self.transform_matrix.shape[0])

        # Identify valid (non-NaN, non-inf, non--inf) elements in the spectrum
        valid_indices = np.isfinite(spectrum[0]) if transpose_needed else np.isfinite(spectrum[:, 0])

        # Optimize the loop with vectorized operations
        if transpose_needed:
            spectrum = spectrum.T

        resampled_spectrum = np.dot(self.transform_matrix[:, valid_indices], spectrum[valid_indices])

        return np.squeeze(resampled_spectrum)


    def srf(self, x, mu, sigma):
        """Spectral Response Function """
        u = (x-mu)/abs(sigma)
        y = (1.0/(np.sqrt(2.0*np.pi)*abs(sigma)))*np.exp(-u*u/2.0)
        if y.sum()==0:
            return y
        else:
            return y/y.sum()


    def spectrumResample(self, idx, wl, wl2, fwhm2=10, fill=False):
        """Resample a spectrum to a new wavelength / FWHM.
        I assume Gaussian SRFs"""

        resampled = np.array(self.srf(wl, wl2[idx], fwhm2[idx]/2.35482))


        return resampled


def spectral_response_function(response_range: np.array, mu: float, sigma: float):
    """Calculate the spectral response function.

    Args:
        response_range: signal range to calculate over
        mu: mean signal value
        sigma: signal variation

    Returns:
        np.array: spectral response function

    """

    u = (response_range - mu) / abs(sigma)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * abs(sigma))) * np.exp(-u * u / 2.0)
    srf = y / y.sum()
    return srf


def resample_spectrum(
    x: np.array, wl: np.array, wl2: np.array, fwhm2: np.array, fill: bool = False, H: np.array = None
) -> np.array:
    """Resample a spectrum to a new wavelength / FWHM.
       Assumes Gaussian SRFs.

    Args:
        x: radiance vector
        wl: sample starting wavelengths
        wl2: wavelengths to resample to
        fwhm2: full-width-half-max at resample resolution
        fill: boolean indicating whether to fill in extrapolated regions

    Returns:
        np.array: interpolated radiance vector

    """
    if H is None:
        H = np.array(
            [
                spectral_response_function(wl, wi, fwhmi / 2.355)
                for wi, fwhmi in zip(wl2, fwhm2)
            ]
        )
        H[np.isnan(H)] = 0

    dims = len(x.shape)
    if fill:
        if dims > 1:
            raise Exception("resample_spectrum(fill=True) only works with vectors")

        x = x.reshape(-1, 1)
        xnew = np.dot(H, x).ravel()
        good = np.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = np.argmin(abs(wl2[good] - wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew
    else:
        # Replace NaNs with zeros
        x[np.isnan(x)] = 0

        # Matrix
        if dims > 1:
            return np.dot(H, x.T).T

        # Vector
        else:
            x = x.reshape(-1, 1)
            return np.dot(H, x).ravel()


def main():
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--output_sixs_wl_file', type=str, default=None)
    parser.add_argument('--output_modtran_wl_file', type=str, default=None)
    parser.add_argument('--irr_file', type=str, default=None)
    args = parser.parse_args()

    output_sixs = None
    output_sixs_wl = None
    if args.output_sixs_wl_file is not None:

        npzf = np.load(args.input_file)
        input_sixs_wl = npzf['sixs_wavelengths']

        output_sixs_wl = np.genfromtxt(args.output_sixs_wl_file)[:, 1]
        output_sixs_fwhm = np.genfromtxt(args.output_sixs_wl_file)[:, 2]
        if np.all(output_sixs_wl < 1000):
            output_sixs_wl *= 1000
            output_sixs_fwhm *= 1000
        
        H = np.array( [ spectral_response_function(input_sixs_wl, wi, fwhmi) for wi, fwhmi in zip(output_sixs_wl, output_sixs_fwhm) ] )
        H[np.isnan(H)] = 0

        input_sixs = npzf['sixs_results'].copy()
        input_solar_irr = npzf['solar_irradiance']
        points = npzf['points']
        point_names = npzf['point_names']

        if args.irr_file is not None:
            irr_wl, irr = np.loadtxt(args.irr_file, comments="#").T
            irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
            input_sixs_irr = np.array(resample_spectrum(irr, irr_wl, input_sixs_wl, input_sixs_wl[1]-input_sixs_wl[0]),dtype=np.float32)

        n_bands = int(input_sixs.shape[1] / len(npzf['keys']))
        for key in range(len(npzf['keys'])):
            input_sixs[:, n_bands * key : n_bands * (key + 1)] = input_sixs[:, n_bands * key : n_bands * (key + 1)] * input_sixs_irr * np.pi / np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))
        
        output_sixs = np.dot(input_sixs, H.T)
        print(output_sixs.shape)
    else:
        output_sixs = npzf['sixs_results']
        output_sixs_wl = npzf['sixs_wavelengths']
    
    output_modtran = None
    output_solar_irr = None
    output_modtran_wl = None
    if args.output_modtran_wl_file is not None:
        npzf = np.load(args.input_file)
        input_modtran_wl = npzf['modtran_wavelengths']

        output_modtran_wl = np.genfromtxt(args.output_modtran_wl_file)[:, 1]
        output_modtran_fwhm = np.genfromtxt(args.output_modtran_wl_file)[:, 2]
        if np.all(output_modtran_wl < 1000):
            output_modtran_wl *= 1000
            output_modtran_fwhm *= 1000

        H = np.array( [ spectral_response_function(input_modtran_wl, wi, fwhmi) for wi, fwhmi in zip(output_modtran_wl, output_modtran_fwhm) ] )
        H[np.isnan(H)] = 0

        input_modtran = npzf['modtran_results'].copy()
        input_solar_irr = npzf['solar_irradiance']
        points = npzf['points']

        if args.irr_file is not None:
            irr_wl, irr = np.loadtxt(args.irr_file, comments="#").T
            irr = irr / 10
            input_modtran_irr = np.array(resample_spectrum(irr, irr_wl, input_modtran_wl, input_modtran_wl[1] - input_modtran_wl[0]),dtype=np.float32)
            output_modtran_irr = np.array(resample_spectrum(irr, irr_wl, output_modtran_wl, output_modtran_fwhm),dtype=np.float32)

        n_bands = int(input_modtran.shape[1] / len(npzf['keys']))
        for key in range(len(npzf['keys'])):
            input_modtran[:, n_bands * key : n_bands * (key + 1)] = input_modtran[:, n_bands * key : n_bands * (key + 1)] * input_modtran_irr * np.pi / np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))

        output_modtran = np.dot(input_modtran, H.T)
        print(output_modtran.shape)


    else:
        output_modtran = npzf['modtran_results']
        output_modtran_irr = npzf['sol_irr']
        output_modtran_wl = npzf['modtran_wavelengths']
    
    
    np.savez(args.output_file, modtran_results=output_modtran, 
                         sixs_results=output_sixs, 
                         points=npzf['points'],
                         sol_irr=output_modtran_irr,
                         sixs_wavelengths=output_sixs_wl,
                         modtran_wavelengths=output_modtran_wl,
                         point_names=npzf['point_names'],  
                         keys=npzf['keys']
                         )



        

if __name__ == "__main__":
    main()
#input_file = "/Users/brodrick/Downloads/emit_emulator_v1.hdf5"
##out_file = "/Users/brodrick/Downloads/emit_emulator_v1_conv01.hdf5"
#out_file = "/Users/brodrick/Downloads/emit_emulator_v1_conv_EMIT_Wavelengths_20220817_clipped.hdf5"
#out_file = "/Users/brodrick/repos/kf/convolved/neon_emulator_v1_conv.hdf5"
#subprocess.call(f"cp {input_file} {out_file}", shell=True)

input_file = "data_202411121/emit_emulator_v2_0.hdf5"
out_file = "data_202411121/emit_emulator_v2_conv_EMIT_Wavelengths_20220817_clipped.hdf5"

input =  h5py.File(input_file, "r")
output = h5py.File(out_file, "w")

def copy_groups(name, obj):
    if isinstance(obj, h5py.Group):
        output.create_group(name)
input.visititems(copy_groups)

def copy_attributes(name, obj):
    if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
        for key, value in obj.attrs.items():
            output[name].attrs[key] = value
input.visititems(copy_attributes)


wl = input["MISCELLANEOUS"]["Wavelengths"][:]
X = input["sample_space"]["sample space"][:,:]
Y_all = input["mod_output/modtran output"][:,:,:]


output_wl = np.genfromtxt('wl/EMIT_Wavelengths_20220817.txt')[:,1:][::-1,:]*1000
subset = np.where((output_wl[:,0] >= wl[0]) & (output_wl[:,0] <= wl[-1]))[0]
output_wl = output_wl[subset,:]
subset = np.where((output_wl[:,0] >= 380) & (output_wl[:,0] <= 2493))[0]
output_wl = output_wl[subset,:]
#output_wl = np.genfromtxt('wl/neon_wl.txt')[:,1:]


print('prepping')
for key, value in input["sample_space"].items():
    output.create_dataset(f"sample_space/{key}", data=value)
for key, value in input["MISCELLANEOUS"].items():
    output.create_dataset(f"MISCELLANEOUS/{key}", data=value)

# overwrite wavelengths
del output["MISCELLANEOUS/Wavelengths"]
output["MISCELLANEOUS/Wavelengths"] = output_wl[:,0]

yn = input['mod_output'].attrs['Products'].tolist()
del input
print('convolving')
##res = resample(wl, np.arange(wl[0], wl[-1], 0.1), 0.1*np.ones(len(np.arange(wl[0], wl[-1], 0.1))))
##res = resample(wl, np.arange(wl[0], wl[-1], 5.0), 5.0*np.ones(len(np.arange(wl[0], wl[-1], 5.0))))
print(output_wl[:,0].shape)
print(Y_all.shape)
print(wl.shape)
res = resample(wl, output_wl[:,0], output_wl[:,1])

yn_out = []
Y_out = []
yn_out.append('path_radiance')
Y_out.append(res.process_data(Y_all[:,yn.index('path_radiance'),:], Y_all.shape[0]))

#yn_out.append('t_down_dir_t_up_dir')
yn_out.append('bi-direct')
Y_out.append(res.process_data(Y_all[:,yn.index('t_down_dir'),:]*\
                              Y_all[:,yn.index('t_up_dir'),:]*\
                              Y_all[:,yn.index('ToA_irrad'),:], Y_all.shape[0]))

#yn_out.append('t_down_dif_t_up_dir')
yn_out.append('hemi-direct')
Y_out.append(res.process_data(Y_all[:,yn.index('t_down_dif'),:]*\
                              Y_all[:,yn.index('t_up_dir'),:]*\
                              Y_all[:,yn.index('ToA_irrad'),:], Y_all.shape[0]))

#yn_out.append('t_down_dir_t_up_dif')
yn_out.append('direct-hemi')
Y_out.append(res.process_data(Y_all[:,yn.index('t_down_dir'),:]*\
                              Y_all[:,yn.index('t_up_dif'),:]*\
                              Y_all[:,yn.index('ToA_irrad'),:], Y_all.shape[0]))

#yn_out.append('t_down_dif_t_up_dif')
yn_out.append('bi-hemi')
Y_out.append(res.process_data(Y_all[:,yn.index('t_down_dif'),:]*\
                              Y_all[:,yn.index('t_up_dif'),:]*\
                              Y_all[:,yn.index('ToA_irrad'),:], Y_all.shape[0]))

yn_out.append('sphalb')
Y_out.append(res.process_data(Y_all[:,yn.index('sphalb_num'),:]/\
                              Y_all[:,yn.index('sphalb_denom'),:], Y_all.shape[0]))

Y_out = np.stack(Y_out, axis=1)


print('writing')
output["mod_output/modtran output"] = Y_out
output['mod_output'].attrs['Products'] = yn_out
del output

print('checking')
input = h5py.File(out_file, "r")
Y_all = input["mod_output/modtran output"][:,:,:]
print(Y_all.shape)
del input


#Y_out = []
#for n in range(Y_all.shape[1]):
#    Y_out.append(res.process_data(Y_all[:,n,:], Y_all.shape[0]))
#
#Y_out = np.stack(Y_out, axis=1)
#
#del input["mod_output/modtran output"]
#input["mod_output/modtran output"] = Y_out
#del input["MISCELLANEOUS/Wavelengths"]
#input["MISCELLANEOUS/Wavelengths"] = output_wl[:,0]
#del input
#input = h5py.File(out_file, "r")
#Y_all = input["mod_output/modtran output"][:,:,:]
#print(Y_all.shape)









