
# This version of the dense_process_beta.py code does not contain the option between 19 and 21 max timepoints, it only will do 19 max timepoints
# This version is outputting hdf5 files with data from all steps in the code into a directory labeled 'dense_process_output_hdf5s' located in each slice directory within a scan

#%%
import numpy as np
import h5py
import math
import time
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
import scipy
import cmath
import statistics as stats
import scipy.signal as sg
import pandas as pd 
import argparse
from functions_MoCo2DProject import *
i = 1j
# cat = which specific folder of files you are using (using whole path when on network as data is nowhere near python code)
cat = '/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/' 
start3 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--recalculate_all", type=str, default='no') # if 'yes', will recalculate all arrays
parser.add_argument("--max_timepoints", type=int, default=19)

args = parser.parse_args()

#~~~~~~~~~~~~~~~~~~ subject to change - variables dependent on specific datasets ~~~~~~~~~~~~~~~~~~#
file_ct = 320 # number of data files to run through (should be # of encodes times the # of frames)
enc_ct = 80
frame_ct = 4
matrix_size = 64 # "image size" essentially, in this case it's a 64x64 pixel image
scan_loc = 16 # scan location? this is apparently a fairly useless number in our case, data is the same across all 16 scan locations
num_frames = 10 # number of frames taken over the course of the cardiac cycle
num_enc = 32 # number of encodes
D_enc = 2.02
full_E = np.matrix([[-1, -1, -1,  1], # encoding matrix required for our number of encodes, (rows,cols)
                    [ 1,  1, -1,  1],
                    [ 1, -1,  1,  1],
                    [-1,  1,  1,  1],
                    [ 1,  1,  1,  1],
                    [-1, -1,  1,  1],
                    [-1,  1, -1,  1],
                    [ 1, -1, -1,  1],
                    [-1, -1, -1,  1], # --> same 8x4 matrix repeated 4 times; leaving the '#' up to delineate the repititions
                    [ 1,  1, -1,  1],
                    [ 1, -1,  1,  1],
                    [-1,  1,  1,  1],
                    [ 1,  1,  1,  1],
                    [-1, -1,  1,  1],
                    [-1,  1, -1,  1],
                    [ 1, -1, -1,  1],
                    [-1, -1, -1,  1], # 
                    [ 1,  1, -1,  1],
                    [ 1, -1,  1,  1],
                    [-1,  1,  1,  1],
                    [ 1,  1,  1,  1],
                    [-1, -1,  1,  1],
                    [-1,  1, -1,  1],
                    [ 1, -1, -1,  1],
                    [-1, -1, -1,  1], #
                    [ 1,  1, -1,  1],
                    [ 1, -1,  1,  1],
                    [-1,  1,  1,  1],
                    [ 1,  1,  1,  1],
                    [-1, -1,  1,  1],
                    [-1,  1, -1,  1],
                    [ 1, -1, -1,  1]])


all_modes = ['ztBFR_noUWPthresh_vTR_detrendtozero', # 0
         'ztBFR_UWPthresh_vTR_detrendtozero', # 1
         'ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp', # 2
         'ztBFR_UWPthresh_vTR_detrendtozero_subB1avg_subB1Interp', # 3
         'ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp'] # 4

modes = all_modes

# modes = ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']

# preproject_dirs = ['dense_motiontest_02141_20240703','dense_motiontest_02146_20240706','dense_motiontest_02152_20240709','dense_motiontest_02153_20240709']
project_dirs = ['dense_motionstandardized_02657_20241025','dense_motionstandardized_02680_20241030','dense_motionstandardized_02692_20241101',
                'dense_motionstandardized_02694_20241101','dense_motionstandardized_02698_20241101','dense_motionstandardized_02710_20241104',
                'dense_motionstandardized_02727_20241107','dense_motionstandardized_02829_20241205','dense_motionstandardized_02997_20250130',
                'dense_motionstandardized_03019_20250204']

for mode in modes:
    print()
    print('\n'+mode+'\n')
    # scan_list = os.listdir(cat)
    scan_list = project_dirs
    for scan in scan_list:
        path_b = cat + scan
        slice_list = os.listdir(path_b)
        slice_list.sort()
        if 'dense_process_output_arrays__19' in slice_list:
            slice_list.remove('dense_process_output_arrays__19')
        if 'dense_process_output_arrays__21' in slice_list:
            slice_list.remove('dense_process_output_arrays__21')
        if 'PCVIPR' in slice_list:
            slice_list.remove('PCVIPR')
        if 'Art_Flow' in slice_list:
            slice_list.remove('Art_Flow')
        print('\n\nworking on the '+scan+' scan now\n')
        
        overall_rr = np.empty(len(slice_list))
        overall_bpm = np.empty(len(slice_list))

        does_it_exist = False

        for number in range(len(slice_list)):
            c_data_folder = slice_list[number]
            current_data_folder = '/' + slice_list[number]
            path_c = path_b + current_data_folder
            os.chdir(path_c)
            y = str(number + 1)
            print('\n\nonto ' + y + ' (' + str(slice_list[number]) + ')')

            path_to_hdf5 = ('/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/'+scan+'/'+str(c_data_folder)+'/dense_process_output_hdf5s_newmasks')

            # if os.path.exists(path_to_hdf5+'/'+mode+'.hdf5'):
            #     does_it_exist = True
            # else:
            #     does_it_exist = False

            # if args.recalculate_all == 'yes':
            #     does_it_exist = False

            if does_it_exist == False:
                #~~~ pulling in all the complex data into a 64x64x16x80 array (64 rows, 64 columns, 16 scan loc, 80 images) ~~~#
                dense_image = np.empty((matrix_size,matrix_size,scan_loc,file_ct),np.csingle)
                for j in range(frame_ct): # going to loop over all 4 averages
                    for i in range(enc_ct): # going to loop over all 80 files

                        filename = ('X_%03d_' % (i)) + ('%03d.dat.complex' % (j))
                        fid = open((path_c + '/' + filename),'r') # opens a single data file
                        raw = np.fromfile(fid,np.csingle) # raw complex data getting read in
                        reshaped_raw = (raw.reshape([scan_loc,matrix_size,matrix_size])).T # reshaping the raw data into a 64x64x16 matrix

                        dense_image[:,:,:,(i+(j*80))] = reshaped_raw

                for i in range(file_ct): #normalizes all the images in scan_loc 3 (does not matter which scan loc)
                    current = np.reshape(dense_image[:,:,2,i],(matrix_size*matrix_size,1))
                    dense_image[:,:,2,i] = ((dense_image[:,:,2,i])/np.amax(current,axis=0))

                pi = math.pi # quickly establishing pi as a value

                images_mag = np.empty((matrix_size,matrix_size,1,file_ct))
                images_mag = abs(dense_image[:,:,0,:])
                images_phase = np.empty((matrix_size,matrix_size,1,file_ct))
                images_phase = np.angle(dense_image[:,:,0,:])

                #%%

                #~~~ creating a matrix of the encodes ~~~#
                encodes = np.empty((matrix_size,matrix_size,num_frames,num_enc),np.csingle); # creating an empty array for the encodes
                for i in range(num_enc):
                    encodes[:,:,:,i] = np.squeeze(dense_image[:,:,0,(i*num_frames):(((i+1)*num_frames))]); # sorting the 80 images by encode


                #~~~ finding the encode with lowest variance and subtracting it ~~~#
                encodes_size = np.array(encodes.shape)

                if mode in modes:
                    encodes_bfr = np.empty((encodes.shape),np.csingle)
                    for i in range(encodes_size[3]):
                        for j in range(encodes_size[2]):
                            encodes_bfr[:,:,j,i] = (encodes[:,:,j,i])*(np.conj(encodes[:,:,0,i]))

                encodes_bfr_size = np.array(encodes_bfr.shape)

                images_bfr = np.empty((matrix_size,matrix_size,(encodes_bfr_size[3]*10)))
                images_bfr_mag = np.empty((matrix_size,matrix_size,(encodes_bfr_size[3]*10)))
                for a in range(encodes_bfr_size[3]):
                    for b in range (num_frames):
                        images_bfr[:,:,((a*10)+b)] = np.angle(encodes_bfr[:,:,b,a])
                        images_bfr_mag[:,:,((a*10)+b)] = abs(encodes_bfr[:,:,b,a])


                phase_enc = np.angle(encodes_bfr)
                phase_enc_size = np.array(phase_enc.shape)


                # os.chdir('/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/'+scan+'/' + str(c_data_folder) + '/_masks')
                # # l_mask = (np.load('eroded_mask_L.npy')) 
                # # r_mask = (np.load('eroded_mask_R.npy')) 
                # l_mask = (np.load('eroded_standardized_left_mask.npy')) 
                # r_mask = (np.load('eroded_standardized_right_mask.npy')) 
                # comb_mask = np.logical_or(l_mask,r_mask) # combined mask for that slice
                # # inner_s_mask = (np.load('hdls_mask_inner.npy'))
                # # outer_s_mask = (np.load('hdls_mask_outer.npy'))
                # # inverse_inner_s_mask = np.invert(inner_s_mask)
                # # skull_mask = np.logical_and(outer_s_mask,inverse_inner_s_mask)
                # # np.save('mask_skull_ring.npy',skull_mask)
                # # inner_b_mask = (np.load('hdls_mask_innerb.npy'))
                # # outer_b_mask = (np.load('hdls_mask_outerb.npy'))
                # outer_b_mask = (np.load('standardized_outer_ring_mask.npy'))
                # # inverse_inner_b_mask = np.invert(inner_b_mask)
                # # brain_mask = np.logical_and(outer_b_mask,inverse_inner_b_mask)
                # # b1_mask = (np.load('eroded_mask_outerb_1.npy'))
                # # b2_mask = (np.load('eroded_mask_outerb_2.npy'))
                # b1_mask = (np.load('standardized_outer_ring_mask_1.npy'))
                # b2_mask = (np.load('standardized_outer_ring_mask_2.npy'))
                # inverse_b1_mask = np.invert(b1_mask)
                # inverse_b2_mask = np.invert(b2_mask)
                # brain1_mask = np.logical_and(outer_b_mask,inverse_b1_mask)
                # brain2_mask = np.logical_and(outer_b_mask,inverse_b2_mask)
                # # np.save('mask_brain_ring.npy',brain_mask)
                # # between_b_mask = (np.load('hdls_mask_betweenb.npy'))
                # # outer_b_mask = (np.load('hdls_mask_outerb.npy'))
                # # inverse_between_b_mask = np.invert(between_b_mask)
                # # thin_brain_mask = np.logical_and(outer_b_mask,inverse_between_b_mask)
                # # np.save('mask_brain_thin_ring.npy',thin_brain_mask)

                ########## new masks section ##########
                os.chdir('/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/'+scan+'/' + str(c_data_folder) + '/_newmasks')
                # os.chdir('/data/data_mrcv/99_LARR/for_caroline/dense_multislice/reconstructed_data/'+scan+'/' + str(c_data_folder) + '/_newmasks')
                l_mask = (np.load('mask_left.npy')) 
                r_mask = (np.load('mask_right.npy')) 
                comb_mask = np.logical_or(l_mask,r_mask)
                outer_b_mask = (np.load('mask_filled_brain_ring.npy'))
                brain1_mask = (np.load('mask_brain_ring.npy'))
                brain2_mask = (np.load('mask_brain_ring_2pixel.npy')) #just so it doesn't throw up an error
           

                #~~~ unwrapping the phase ~~~#
                unwrap_ph = np.empty(phase_enc.shape)
                unwrap_ph_forward = np.empty(phase_enc.shape)
                uwp_diff = []

                for i in range(phase_enc_size[3]):
                    ph = phase_enc[:,:,:,i]
                    unwrap_ph_forward[:,:,:,i] = np.unwrap(ph,axis=2)
                    unwrap_ph[:,:,:,i] = unwrap_ph_forward[:,:,:,i]
                    if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                        for j in range(phase_enc_size[2]-1):
                            first = np.squeeze(unwrap_ph_forward[:,:,j,i]) 
                            second = np.squeeze(unwrap_ph_forward[:,:,j+1,i]) 
                            if (np.abs(second[l_mask] - first[l_mask]) > (3*pi/4)).any(): 
                                if i not in uwp_diff:
                                    uwp_diff.append(i)
                            if (np.abs(second[r_mask] - first[r_mask]) > (3*pi/4)).any(): 
                                if i not in uwp_diff:
                                    uwp_diff.append(i)

                if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    print('\n3pi/4 threshold:')
                    print('encodes flagged using eroded mask: '+str(uwp_diff))

                unwrap_ph_size = np.array(unwrap_ph.shape)
                images_uwp = np.full((matrix_size,matrix_size,(encodes_bfr_size[3]*10)),np.nan)
                for d in range(encodes_bfr_size[3]):
                    for e in range(10):
                        images_uwp[:,:,((d*10)+e)] = unwrap_ph[:,:,e,d]

                if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    phase_fits = np.empty(unwrap_ph.shape)
                    fit_coeffs = np.empty((3,unwrap_ph.shape[2],unwrap_ph.shape[3]))
                    subtracted_phase = np.full((unwrap_ph.shape),np.nan)
                    for z in range(unwrap_ph_size[3]):
                        for w in range(unwrap_ph_size[2]):
                            full_data_to_fit = unwrap_ph[:,:,w,z]
                            if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']:
                                z_rec, ic_rec, slx_rec, sly_rec = complex_regression2Dj_new(full_data_to_fit,brain1_mask,30)
                            if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                                z_rec, ic_rec, slx_rec, sly_rec = complex_regression2Dj_new(full_data_to_fit,brain2_mask,30)
                            fit_coeffs[:,w,z] = [slx_rec,sly_rec,ic_rec]
                            phase_fits[:,:,w,z] = z_rec[:,:]
                            # sub_unwrap_ph[:,:,j,i] = refr_unwrap_ph[:,:,j,i] - z_rec[:,:]
                            for a in range(unwrap_ph.shape[0]):
                                for b in range(unwrap_ph.shape[1]):
                                    # if outer_b_mask[a,b] == True:
                                    i = 1j
                                    subtracted_phase[a,b,w,z] = np.angle(cmath.exp(i*(unwrap_ph[a,b,w,z]))*cmath.exp(-i*(z_rec[a,b])))
                    unwrap_ph[:,:,:,:] = subtracted_phase[:,:,:,:]

                    images_comp = np.full((matrix_size,matrix_size,(encodes_bfr_size[3]*10)),np.nan)
                    for k in range(encodes_bfr_size[3]):
                        for l in range(10):
                            # print(images_comp.shape)
                            # print(subtracted_phase.shape)
                            images_comp[:,:,((k*10)+l)] = subtracted_phase[:,:,l,k]
                            
                # if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']:
                #     images_uwp = np.full((matrix_size,matrix_size,(encodes_bfr_size[3]*10)),np.nan)
                #     for k in range(encodes_bfr_size[3]):
                #         for l in range(10):
                #             images_uwp[:,:,((k*10)+l)] = unwrap_ph[:,:,k,l]
                

             #################### heartrate section start - commenting out to run the motion scans through it ####################
                
                known_time_points = np.array(range(0,19,2))
                time_points_to_interpolate = np.array(range(1,19,2))
                #print(known_time_points)
                #print(time_points_to_interpolate)


                output_ph = np.empty((unwrap_ph.shape[0],unwrap_ph.shape[1],((2*unwrap_ph.shape[2])-1),unwrap_ph.shape[3]))
                print(output_ph.shape)
                for a in range(unwrap_ph.shape[0]):
                    for b in range(unwrap_ph.shape[1]):
                        for c in range(unwrap_ph.shape[3]):
                            actual_phase = unwrap_ph[a,b,:,c]
                            interpolated_phase = np.array(np.interp(time_points_to_interpolate, known_time_points, unwrap_ph[a,b,:,c]))
                            output_ph[a,b,::2,c] = actual_phase[:]
                            output_ph[a,b,1::2,c] = interpolated_phase[:]
                unwrap_ph = output_ph


                os.chdir(path_c)
                print(path_c)

                ScanArch_list = glob.glob('*.h5.hdr')
                print(ScanArch_list)
                ScanArch_hdr = ScanArch_list[0]
                num = "imagehead.tr="
                with open(ScanArch_hdr) as search:
                    for line in search:
                        line = line.rstrip()  # remove '\n' at end of line
                        if num in line:
                            print(line)
                            tr_amount_times1000 = int(line.replace('imagehead.tr=',''))
                            tr_amount = tr_amount_times1000/1000
                            #print(tr_amount)


                #~~~ heartrate section, filtering out images running into next heartbeat ~~~#
                DataFile = glob.glob('PPGData_pcvipr_*')
                print(DataFile)

                if len(DataFile) != 0.0:
                    ppg_datafile = open(path_c+'/'+str(DataFile[0].replace('.md5sum','')),'r')
                    ppg_data = np.array(ppg_datafile.read().split('\n '))
                    ppg_data = ppg_data.astype(np.float64)
                    ppg_time = np.array(np.arange(-30000,((np.int32(ppg_data.shape[0])*10)-30000),10)) #ppg_data.shape[0])) # goes from -30,000 to however long the scan lasted 
                    ppg_datafile.close()
                    ppg_triggers_altered = []
                    trigger = False
                    for item in range(len(ppg_data)):
                        if (ppg_data[item] > 250.0):
                            if trigger == False:
                                ppg_triggers_altered.append(ppg_time[item])
                                trigger = True #setting trigger to true because we went over 300
                            #this^ no longer is activated now for all proceeding values above 300 because the trigger is true now
                            # else:
                            #     continue
                        if (ppg_data[item] < 20.0):
                            if trigger == True:
                                trigger = False
                            # else:
                            #     continue
                        # else:
                        #     continue
                    ppg_triggers_altered = np.array(ppg_triggers_altered)

                TriggerFile = glob.glob('PPGTrig_pcvipr_*')
                if len(TriggerFile) != 0.0:
                    ppg_triggerfile = open(path_c+'/'+str(TriggerFile[0].replace('.md5sum','')),'r')
                    ppg_triggers = np.array(ppg_triggerfile.read().split('\n'))
                    ppg_triggers = ppg_triggers[:-1].astype(np.int32)
                    ppg_triggers = (ppg_triggers*10)-30000 # changes the trigger moments from index values to increments of 10ms, with index "1" being -30,000ms
                    ppg_triggerfile.close()
                    # ppg data time should be -30 + 10ms*whatever index #########

                if len(DataFile) != 0.0:
                    rr_windows = np.empty(np.squeeze(ppg_triggers.shape)-1)
                    for element in range(np.squeeze(ppg_triggers.shape)-1):
                        rr_windows[element] = ppg_triggers[element+1] - ppg_triggers[element] #in index amounts
                    rr_windows = rr_windows*1 #in milliseconds
                    RMSSD_rr_windows = rmssd(rr_windows)
                if len(DataFile) != 0.0:
                    rr_windows_corrected = np.empty(np.squeeze(ppg_triggers_altered.shape)-1)
                    for element_corr in range(np.squeeze(ppg_triggers_altered.shape)-1):
                        rr_windows_corrected[element_corr] = ppg_triggers_altered[element_corr+1] - ppg_triggers_altered[element_corr] #in index amounts
                    rr_windows_corrected = rr_windows_corrected*1 #in milliseconds
                    RMSSD_rr_windows_corrected = rmssd(rr_windows_corrected)

                    mean_rr = np.mean(rr_windows)
                    median_rr = np.median(rr_windows) # in milliseconds
                    median_bpm = (1/median_rr)*(60000) # in beats per minute

                    overall_rr[number] = median_rr
                    overall_bpm[number] = median_bpm

                if len(DataFile) == 0.0: # we have at least one case that doesn't appear to have PPGData or PPGTrigger files, so this is for them
                    print('error, no DataFile')
                    overall_rr[number] = np.nan
                    overall_bpm[number] = np.nan
                    rr_windows = np.empty(1)
                    RMSSD_rr_windows = np.empty(1)
                    rr_windows_corrected = np.empty(1)
                    RMSSD_rr_windows_corrected = np.empty(1)


                print('\nmedian rr window: '+str(overall_rr[number]))
                print('median bpm window: '+str(overall_bpm[number]))
                print('unwrap array shape: '+str(unwrap_ph.shape))


                time_counts = 19
                unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))

                # if mode not in ['origBFR_noUWPthresh_fTR_scipydetrend','origBFR_UWPthresh_fTR_scipydetrend','origBFR_UWPthresh_fTR_detrendtozero']:
                tot_tr_amount = 9*tr_amount
                tr_steps_to_subtract = tr_amount/2
                print('total tr amount (tr amount times 9): '+str(tot_tr_amount))
                #print(tr_steps_to_subtract)
                if overall_rr[number]<=tot_tr_amount:
                    if (tot_tr_amount-(2*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-tr_steps_to_subtract):
                        time_counts = 18
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(3*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(2*tr_steps_to_subtract)):
                        time_counts = 17
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(4*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(3*tr_steps_to_subtract)):
                        time_counts = 16
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(5*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(4*tr_steps_to_subtract)):
                        time_counts = 15
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(6*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(5*tr_steps_to_subtract)):
                        time_counts = 14
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(7*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(6*tr_steps_to_subtract)):
                        time_counts = 13
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(8*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(7*tr_steps_to_subtract)):
                        time_counts = 12
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    if (tot_tr_amount-(9*tr_steps_to_subtract))<=overall_rr[number]<(tot_tr_amount-(8*tr_steps_to_subtract)):
                        time_counts = 11
                        unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    # if overall_rr[number]<(tot_tr_amount-(9*tr_steps_to_subtract)):
                    #     time_counts = 10
                    #     unwrap_ph = np.reshape(unwrap_ph[:,:,:time_counts,:],(unwrap_ph_size[0],unwrap_ph_size[1],time_counts,unwrap_ph_size[3]))
                    else:
                        time_counts = 19
                else:
                    time_counts = 19

                    # print('unwrap array shape after heartrate filtering: '+str(unwrap_ph.shape)+'\n')
                    #print(time_counts)



             #################### heartrate section end - commenting out to run the motion scans through it ####################

                unwrap_ph_size = np.array(unwrap_ph.shape)
                print('unwrap array shape after heartrate filtering: '+str(unwrap_ph.shape)+'\n')

                # if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']:
                #     bub_unwrap_ph = np.full(unwrap_ph.shape,np.nan)
                #     phase_fits = np.empty(unwrap_ph.shape)
                #     fit_coeffs = np.empty((3,unwrap_ph.shape[2],unwrap_ph.shape[3]))
                #     for i in range(unwrap_ph_size[3]):
                #         for j in range(unwrap_ph_size[2]):
                #             z_rec, ic_rec, slx_rec, sly_rec = complex_regression2D(unwrap_ph[:,:,j,i],brain1_mask,50)
                #             fit_coeffs[:,j,i] = [slx_rec,sly_rec,ic_rec]
                #             phase_fits[:,:,j,i] = z_rec[:,:]
                #             # sub_unwrap_ph[:,:,j,i] = refr_unwrap_ph[:,:,j,i] - z_rec[:,:]
                #             for a in range(unwrap_ph.shape[0]):
                #                 for b in range(unwrap_ph.shape[1]):
                #                     # if outer_b_mask[a,b] == True:
                #                     bub_unwrap_ph[a,b,j,i] = np.angle(cmath.exp(i*(unwrap_ph[a,b,j,i]))*cmath.exp(-i*(z_rec[a,b])))
                #     unwrap_ph = bub_unwrap_ph

                
                #~~~ linear detrending, to help remove bulk motion ~~~#
                voxel_tseries = np.empty(unwrap_ph.shape)
                voxel_detrend = np.empty(unwrap_ph.shape)
                detrend_unwrap_ph = np.empty(unwrap_ph.shape)
                for z in range(unwrap_ph_size[3]):
                    for i in range(unwrap_ph_size[0]):
                        for j in range(unwrap_ph_size[1]):
                            voxel_tseries = np.squeeze(unwrap_ph[i,j,:,z])
                            slope = (voxel_tseries[-1] - voxel_tseries[0])/(len(voxel_tseries)-1)
                            points_to_subtract = slope*range(len(voxel_tseries))
                            voxel_detrend = voxel_tseries - points_to_subtract
                            voxel_detrend_reshape = voxel_detrend.reshape([1,1,unwrap_ph_size[2]])
                            detrend_unwrap_ph[i,j,:,z] = voxel_detrend_reshape
                #unwrap_ph = detrend_unwrap_ph

                images_dtr = np.empty(unwrap_ph.shape)
                for a in range(unwrap_ph_size[3]):
                    for b in range (unwrap_ph_size[2]):
                        images_dtr[:,:,b,a] = detrend_unwrap_ph[:,:,b,a]


                #~~~ creating a reference at the first time point and subtracting it ~~~#
                refr_unwrap_ph = np.empty(unwrap_ph.shape)
                for i in range(unwrap_ph_size[3]):
                    refr = detrend_unwrap_ph[:,:,0,i]
                    for j in range(unwrap_ph_size[2]):
                        current_encode = detrend_unwrap_ph[:,:,j,i]
                        refr_unwrap_ph[:,:,j,i] = current_encode - refr
                refr_unwrap_ph_size = np.array(refr_unwrap_ph.shape)

                images_ref = np.empty(unwrap_ph.shape)
                for a in range(unwrap_ph_size[3]):
                    for b in range (unwrap_ph_size[2]):
                        images_ref[:,:,b,a] = refr_unwrap_ph[:,:,b,a]


                # if mode in ['ztBFR_UWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_UWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp']:#,'ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']:#,'ztBFR_vTR_detrendtozero_subB1Interp_UWPthreshloop']:
                #     fit_coeffs = np.empty((3,refr_unwrap_ph.shape[2],refr_unwrap_ph.shape[3]))
                #     phase_fits = np.empty(refr_unwrap_ph.shape)


                #~~~ subtraction section - to remove bulk motion ~~~#
                sub_unwrap_ph = np.full(refr_unwrap_ph.shape,np.nan)
                # phase_fits = np.empty(refr_unwrap_ph.shape)
                if mode in ['ztBFR_UWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_UWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp']:#,'ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp']:#,'ztBFR_vTR_detrendtozero_subB1Interp_UWPthreshloop']:
                    fit_coeffs = np.empty((3,refr_unwrap_ph.shape[2],refr_unwrap_ph.shape[3]))
                    phase_fits = np.empty(refr_unwrap_ph.shape)
                    for i in range(refr_unwrap_ph_size[3]):
                        for j in range(refr_unwrap_ph_size[2]):
                            if mode in ['ztBFR_UWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp']:#,'ztBFR_vTR_detrendtozero_subB1Interp_UWPthreshloop']:
                                fit_array, coeffs, intercept = poly_regression2D(refr_unwrap_ph[:,:,j,i],brain1_mask,1)
                                fit_coeffs[:,j,i] = [coeffs[0],coeffs[1],intercept]
                                phase_fits[:,:,j,i] = fit_array[:,:]
                                sub_unwrap_ph[:,:,j,i] = refr_unwrap_ph[:,:,j,i] - fit_array[:,:]
                            if mode in ['ztBFR_UWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp']:
                                fit_array, coeffs, intercept = poly_regression2D(refr_unwrap_ph[:,:,j,i],brain2_mask,1)
                                fit_coeffs[:,j,i] = [coeffs[0],coeffs[1],intercept]
                                phase_fits[:,:,j,i] = fit_array[:,:]
                                sub_unwrap_ph[:,:,j,i] = refr_unwrap_ph[:,:,j,i] - fit_array[:,:]

                if mode in ['ztBFR_UWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    sub_unwrap_ph[:,:,:,:] = refr_unwrap_ph[:,:,:,:]

                sub_unwrap_ph_size = np.array(sub_unwrap_ph.shape)
                
                images_sub = np.full((sub_unwrap_ph.shape),np.nan)
                for a in range(sub_unwrap_ph_size[3]):
                    for b in range (sub_unwrap_ph_size[2]):
                        images_sub[:,:,b,a] = sub_unwrap_ph[:,:,b,a]



                # switching the names of the array back to the 'ref' convention, so that don't need to change all the variable names below
                ref_unwrap_ph = np.empty(sub_unwrap_ph.shape)
                ref_unwrap_ph[:,:,:,:] = sub_unwrap_ph[:,:,:,:]
                ref_unwrap_ph_size = np.array(ref_unwrap_ph.shape)


                full_E_size = full_E.shape
                if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    EI = np.linalg.pinv(full_E)
                if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    restricted_E = np.delete(full_E,uwp_diff,0)
                    EI = np.linalg.pinv(restricted_E)

                    restricted_ref_unwrap_ph = np.delete(ref_unwrap_ph,uwp_diff,3)
                    restricted_ref_unwrap_ph_size = np.array(restricted_ref_unwrap_ph.shape)
                

                scale = (math.sqrt(3)/pi)*(1/D_enc)

                if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    dx = np.zeros((ref_unwrap_ph_size[0],ref_unwrap_ph_size[1],ref_unwrap_ph_size[2])); #size(E,1)
                    dy = dx
                    dz = dx

                    #print('using the restricted encoding matrix')
                    for i in range(ref_unwrap_ph_size[3]):
                        dx = dx + scale*EI[0,i]*ref_unwrap_ph[:,:,:,i]
                        dy = dy + scale*EI[1,i]*ref_unwrap_ph[:,:,:,i]
                        dz = dz + scale*EI[2,i]*ref_unwrap_ph[:,:,:,i]



                if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                    dx = np.zeros((restricted_ref_unwrap_ph_size[0],restricted_ref_unwrap_ph_size[1],restricted_ref_unwrap_ph_size[2])); #size(E,1)
                    dy = dx
                    dz = dx

                    for i in range(restricted_ref_unwrap_ph_size[3]):
                        dx = dx + scale*EI[0,i]*restricted_ref_unwrap_ph[:,:,:,i]
                        dy = dy + scale*EI[1,i]*restricted_ref_unwrap_ph[:,:,:,i]
                        dz = dz + scale*EI[2,i]*restricted_ref_unwrap_ph[:,:,:,i]



                # df = pd.DataFrame({"area":[],"avg_mag":[],"full_mag":[],"stdv_mag":[]})
                diffs = np.full((((dx[:,:,:]).shape[2]),((dx[:,:,:]).shape[2])),np.nan) # 10x10
                max_mag = np.full((((dx[:,:,:]).shape[0]),((dx[:,:,:]).shape[1])),np.nan) # 64x64
                diffs_output = np.full((((dx[:,:,:]).shape[2]),((dx[:,:,:]).shape[2]),((dx[:,:,:]).shape[0]),((dx[:,:,:]).shape[1])),np.nan) # 10x10x64x64
                disp_mag = np.full((((dx[:,:,:]).shape[0]),((dx[:,:,:]).shape[1]),((dx[:,:,:]).shape[2])),np.nan)
                # diffs_df = pd.DataFrame({"diffs_output":[]})

                for c in range((dx[:,:,:]).shape[0]): #in range 64
                    for d in range((dx[:,:,:]).shape[1]): #in range 64 , so for each pixel in the image
                        if outer_b_mask[c,d] == True: # is that pixel contained within the mask; if yes, then find the max magnitude of displacement across time points
                            for a in range((dx[:,:,:]).shape[2]): #in range 10 (or more for hrt stuff)
                                for b in range((dx[:,:,:]).shape[2]): #in range 10

                                    diffs[a,b] = np.sqrt(((((dx[c,d,a])-(dx[c,d,b]))**2)+(((dy[c,d,a])-(dy[c,d,b]))**2)+(((dz[c,d,a])-(dz[c,d,b]))**2)))
                                    diffs_output[a,b,c,d] = diffs[a,b]

                            max_mag[c,d] = np.nanmax((diffs[:,:]))

                        # if outer_b_mask[c,d] == True:
                        for a in range((dx[:,:,:]).shape[2]):
                            disp_mag[c,d,a] = np.sqrt((((dx[c,d,a])**2)+((dy[c,d,a])**2)+((dz[c,d,a])**2)))

                avg_max_mag = np.nanmean((max_mag[comb_mask]),axis=None)

                dx_avgs = np.zeros((ref_unwrap_ph_size[2]))
                dy_avgs = np.zeros((ref_unwrap_ph_size[2]))
                dz_avgs = np.zeros((ref_unwrap_ph_size[2]))
                disp_avgs = np.zeros((ref_unwrap_ph_size[2]))
                for timepoint in range(ref_unwrap_ph_size[2]):
                    dx_image = dx[:,:,timepoint]
                    dy_image = dy[:,:,timepoint]
                    dz_image = dz[:,:,timepoint]
                    disp_image = disp_mag[:,:,timepoint]
                    dx_avgs[timepoint] = np.mean(dx_image[comb_mask])
                    dy_avgs[timepoint] = np.mean(dy_image[comb_mask])
                    dz_avgs[timepoint] = np.mean(dz_image[comb_mask])
                    disp_avgs[timepoint] = np.nanmean(disp_image[comb_mask])


                # path_to_hdf5 = ('/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/'+scan+'/'+str(c_data_folder)+'/dense_process_output_hdf5s')
                path_to_hdf5 = ('/data/data_mrcv/99_LARR/for_caroline/dense_1.0_motion/reconstructed_data/'+scan+'/'+str(c_data_folder)+'/dense_process_output_hdf5s_newmasks')
                # path_to_hdf5 = ('/data/data_mrcv/99_LARR/for_caroline/dense_multislice/reconstructed_data/'+scan+'/'+str(c_data_folder)+'/dense_process_output_hdf5s_newmasks')
                if not os.path.exists(path_to_hdf5):
                    os.makedirs(path_to_hdf5)

                if os.path.exists(path_to_hdf5+'/'+mode+'.hdf5'):
                    os.remove(path_to_hdf5+'/'+mode+'.hdf5')

                os.chdir(path_to_hdf5)

                with h5py.File(mode + '.hdf5','a') as f:
                    f.create_dataset('images_mag', data=images_mag)
                    f.create_dataset('images_phase', data=images_phase)
                    f.create_dataset('images_bfr', data=images_bfr) #this is the bfr phase data
                    f.create_dataset('images_bfr_mag', data=images_bfr_mag) #this is the bfr magnitude data
                    f.create_dataset('images_uwp', data=images_uwp)
                    if mode in ['ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                        f.create_dataset('images_comp', data=images_comp)
                    f.create_dataset('tr_amount', data=tr_amount)
                    f.create_dataset('rr_windows', data=rr_windows)
                    f.create_dataset('RMSSD_rr_windows', data=RMSSD_rr_windows)
                    f.create_dataset('rr_windows_corrected', data=rr_windows_corrected)
                    f.create_dataset('RMSSD_rr_windows_corrected', data=RMSSD_rr_windows_corrected)
                    f.create_dataset('images_dtr', data=images_dtr)
                    f.create_dataset('images_ref', data=images_ref)
                    f.create_dataset('images_sub', data=images_sub)
                    if mode in ['ztBFR_UWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_UWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']: #,'ztBFR_vTR_detrendtozero_subB1Interp_UWPthreshloop'
                        f.create_dataset('phase_fits', data=phase_fits)
                        f.create_dataset('fit_coeffs', data=fit_coeffs)
                    if mode not in ['ztBFR_noUWPthresh_vTR_detrendtozero','ztBFR_noUWPthresh_vTR_detrendtozero_subB1avg_subB1Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2avg_subB2Interp','ztBFR_noUWPthresh_vTR_detrendtozero_subB1ComplexInterp','ztBFR_noUWPthresh_vTR_detrendtozero_subB2ComplexInterp']:
                        f.create_dataset('uwp_diff', data=uwp_diff)
                    f.create_dataset('dx', data=dx)
                    f.create_dataset('dx_avgs', data=dx_avgs)
                    f.create_dataset('dy', data=dy)
                    f.create_dataset('dy_avgs', data=dy_avgs)
                    f.create_dataset('dz', data=dz)
                    f.create_dataset('dz_avgs', data=dz_avgs)
                    f.create_dataset('disp', data=disp_mag)
                    f.create_dataset('disp_avgs', data=disp_avgs)
                    f.create_dataset('dmax', data=max_mag)
                    f.create_dataset('dmax_avgs', data=avg_max_mag)

                print('Calculations output to dense_1.0_motion/reconstructed_data/'+scan+'/'+str(c_data_folder)+'/dense_process_output_hdf5s_newmasks/'+mode+'.hdf5\n')
                
            if does_it_exist == True:
                print('An hdf5 file already exists for '+mode+' in '+scan+'/'+str(c_data_folder)+'.')
                print('Moving onto next one.\n')
                continue


end3 = time.time()
# print("WITH NUMBA TIME END")
print('my version')

length3 = end3 - start3
if length3 > 120.0:
    length3 = length3/60.0
    print('post-numba, it took '+str(length3)+' minutes to do 320 images')
else:
    print('post-numba, it took '+str(length3)+' seconds to do 320 images')


    