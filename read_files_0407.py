import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects as rso
from skimage.transform import probabilistic_hough_line as phl
from pyhrv.hrv import hrv
import warnings

warnings.filterwarnings('ignore')


def simple_show(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
    plt.close()


def filtering_by_std(image, axis=0, threshold=25, upper=True, minimum_pix=500):
    image = np.std(image, axis=axis)
    if upper:
        image = image > threshold
    elif not upper:
        image = image < threshold
    image = rso(image, minimum_pix)
    return image


def abstracter(image, x_range=0, x_screen=0, y_screen=0):
    HR_raw = []
    x_ = np.arange(0, x_range, 1)
    for x in x_:
        y_idx = []
        for y in np.where(image[:, x] == 1)[0]:
            if all([x < x_screen, y < y_screen]):
                continue
            y_idx.append(y)
        if len(y_idx) > 100:
            y_idx = []

        if y_idx:
            mean = np.mean(y_idx)
        else:
            mean = np.nan
        HR_raw.append(mean)
    return HR_raw


def scaler(data, h_pix, h_size, baseline=0, cut=False):
    HR = []
    for i in data:
        if np.isnan(i):
            HR.append(None)
            continue
        value = ((h_pix-int(i))/h_pix*h_size) + baseline
        if cut is True:
            value = round(value, 4)
        HR.append(value)
    return HR


base_path = pathlib.Path('data_preprocessed')
data_type = 'twi'

# fetal bounding box
xl, xr, fhr_yt, fhr_yb = 0, 1612, 0, 475
uc_yt, uc_yb = 550, 835



length_list = []
avg_length = 1100
pid_list = set()

for i in base_path.glob('*.png'):
    name = i.stem.split('(')[0]
    pid_list.add(name)

print('Patient id: ', pid_list)

hrv_data = None
hrv_data2 = None

for p in pid_list:
    l = []
    l_mhr = []
    l_uc = []
    for idx, i in enumerate(base_path.glob(f'{p}*.png')):
        print('file number: ', i.stem.split('(')[-1][:-1])
        series = int(i.stem.split('(')[-1][:-1])
        color = imread(i)
        gray = rgb2gray(color)
        # simple_show(color)
        # simple_show(gray)

        reversed_gray = 1 - gray
        # simple_show(reversed_gray)

        dd = np.ones(reversed_gray.shape)
        # simple_show(dd)

        dd[reversed_gray < 0.01] = 0
        fhr_box = dd[fhr_yt:fhr_yb, xl:xr]
        uc_box = dd[uc_yt:uc_yb, xl:xr]
        # simple_show(fhr_box)
        # simple_show(uc_box)

        angle = [0, np.pi / 2]
        v_angle = angle[:-1]  # ????????? ??????...
        h_angle = angle[1:]
        v_theta = np.array(v_angle, dtype='double')
        h_theta = np.array(h_angle, dtype='double')
        # threshold ????????? linewidth ??? ????????? ????????? ???????????? ?????? ?????????.
        # line_length ????????? ??? line??? ????????? ???????????? ?????? ?????????.
        v_phl = phl(fhr_box, threshold=150, line_length=100, theta=v_theta)
        h_phl = phl(fhr_box, threshold=200, line_length=200, theta=h_theta)
        h_phl_uc = phl(uc_box, threshold=200, line_length=200, theta=h_theta)

        x_values = [x[0][0] for x in v_phl]
        y_values = [y[0][1] for y in h_phl]
        y_values_uc = [y_uc[0][1] for y_uc in h_phl_uc]

        # print(np.array(x_values).min(),np.array(x_values).max())
        xl_new = xl + np.array(x_values).min()
        xr_new = xl + np.array(x_values).max()
        fhr_yt_new = fhr_yt + np.array(y_values).min()
        fhr_yb_new = fhr_yt + np.array(y_values).max()
        uc_yt_new = uc_yt + np.array(y_values_uc).min()
        uc_yb_new = uc_yt + np.array(y_values_uc).max()

        # print(xl_new, xr_new, fhr_yt_new, fhr_yb_new, uc_yt_new, uc_yb_new)
        length = xr_new-xl_new
        length_list.append(length)
        # simple_show(dd[fhr_yt_new:fhr_yb_new, xl_new:xr_new], figsize=(10,5))
        # simple_show(dd[uc_yt_new:uc_yb_new, xl_new:xr_new], figsize=(10,5))

        fhr_box_color = color[fhr_yt_new:fhr_yb_new, xl_new:xr_new]
        mhr_box_color = color[fhr_yt_new:fhr_yb_new, xl_new:xr_new]     # MHR ?????? ?????? (3/14)
        # mhr_box_color = color[250:470, xl_new:xr_new]     # MHR ?????? ?????? (3/24) >>> Twin ????????? ??? ?????????... 

        uc_box_color = color[uc_yt_new:uc_yb_new, xl_new:xr_new]
        # simple_show(color[fhr_yt_new:fhr_yb_new, xl_new:xr_new], figsize=(10,5))
        # simple_show(color[uc_yt_new:uc_yb_new, xl_new:xr_new], figsize=(10,5))

        if data_type == 'twin':       # Red & Blue Lines... (Twin Fetal)
            # color : threhold??? ???????????? ?????? ????????? ??????. (?????? ??? ????????? ??????)
            # 800 ??? ????????? ????????? ????????? ????????? ??????.
            fhr_std = filtering_by_std(fhr_box_color, axis=2, threshold=20, upper=True, minimum_pix=50)
            fhr_blue = (fhr_box_color[:, :, 0] < 100) & (
                fhr_box_color[:, :, 1] < 100) & (fhr_box_color[:, :, 2] >= 140)
            HR_raw = np.all([fhr_std, fhr_blue], axis=0)   # Blue line

        else:       # Only Red Lines... (Not Twin Fetal)
            fhr_std = filtering_by_std(
                fhr_box_color, axis=2, threshold=25, upper=True, minimum_pix=50)
            fhr_red = np.sum(fhr_box_color, axis=2) < 800     # TJ's Codes... (Use Pixel Std)
            # FHR RGB value >>>  center (R,G,B) : (179, 59, 70),  edge (R,G,B) = (188, 139, 142)           2 # HJ's Codes... (Use Pixel Range)  (R,G,B) = (191, 92, 97)
            fhr_red = (fhr_box_color[:, :, 0] > 161) & (fhr_box_color[:, :, 0] < 221) \
                & (fhr_box_color[:, :, 1] > 62) & (fhr_box_color[:, :, 1] < 110) \
                & (fhr_box_color[:, :, 2] > 60) & (fhr_box_color[:, :, 2] < 127)

            HR_raw = np.all([fhr_std, fhr_red], axis=0)   # FHR lines....

        # MHR RBG value >>> center (R,G,B) : (197, 119, 200) , edge (R,G,B) : (225,200,230)    # MHR ?????? (3/14)
        mhr_std = filtering_by_std(mhr_box_color, axis=2, threshold=15, upper=True, minimum_pix=50)
        mhr_purple = (mhr_box_color[:, :, 0] > 170) & (mhr_box_color[:, :, 0] < 230) \
            & (mhr_box_color[:, :, 1] > 100) & (mhr_box_color[:, :, 1] < 150) \
            & (mhr_box_color[:, :, 2] > 180) & (mhr_box_color[:, :, 2] < 230)

        # MHR lines....     # MHR ?????? (3/14)
        MHR_raw = np.all([mhr_std, mhr_purple], axis=0)

        # simple_show(fhr_std, figsize=(10, 5))
        # simple_show(fhr_red, figsize=(10, 5))
        # simple_show(mhr_purple, figsize=(10, 5))
        # simple_show(MHR_raw, figsize=(10, 5))
        # simple_show(HR_raw, figsize=(10, 5))

        # HR_raw = abstracter(HR_raw, x_range=avg_length-1, x_screen=0, y_screen=0)
        HR_raw = abstracter(HR_raw, x_range=length-1, x_screen=0, y_screen=0)
        HR = scaler(HR_raw, fhr_yb_new-fhr_yt_new, 210,
                    baseline=30, cut=False)  # 142ppi

        MHR_raw = abstracter(MHR_raw, x_range=length-1, x_screen=0, y_screen=0)
        MHR = scaler(MHR_raw, fhr_yb_new-fhr_yt_new, 210, baseline=30, cut=False)  # 142ppi
        # ''' Twin ????????? ?????? ?????????... '''
        # MHR = scaler(MHR_raw, fhr_yb_new-fhr_yt_new, 60, baseline=30, cut=False)  

        # print(HR)
        # plt.figure(figsize=(10,5))        # FHR
        # plt.xlim([0, avg_length])
        # plt.ylim([30, 240])
        # plt.plot(HR)
        # plt.show()
        # plt.close()

        # plt.figure(figsize=(10,5))        # MHR
        # plt.xlim([0, avg_length])
        # plt.ylim([30, 240])
        # plt.plot(MHR)
        # plt.show()

        l.append((series, HR))
        l_mhr.append((series, MHR))
        uc_black = np.min(uc_box_color, axis=2) < 100
        # simple_show(uc_black, figsize=(10, 5))

        # UC_raw = abstracter(uc_black, x_range=avg_length-1, x_screen=0, y_screen=0) #142ppi
        UC_raw = abstracter(uc_black, x_range=length-1,
                            x_screen=0, y_screen=0)  # 142ppi
        UC = scaler(UC_raw, uc_yb_new-uc_yt_new, 100,
                    baseline=0, cut=False)  # 142ppi
        # plt.figure(figsize=(10,5))
        # plt.xlim([0, avg_length])
        # plt.ylim([0, 100])
        # plt.plot(UC)
        # plt.show()
        # plt.close()
        l_uc.append((series, UC))

    l.sort(key=lambda x: x[0])
    l_mhr.sort(key=lambda x: x[0])
    l_uc.sort(key=lambda x: x[0])
    fhr = []
    mhr = []

    print('fhr')
    for f in l:
        fhr.extend(f[1])
        print(len(fhr))

    print('mhr')

    for k in l_mhr:
        mhr.extend(k[1])
        print(len(mhr))

    # plt.figure(figsize=(25, 3))
    # plt.xlim([0, avg_length*len(l)])
    # plt.ylim([30, 240])
    # plt.plot(fhr)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(25, 3))
    # plt.xlim([0, avg_length*len(l_mhr)])
    # plt.ylim([30, 240])
    # plt.plot(mhr)
    # plt.show()

    uc = []
    for u in l_uc:
        uc.extend(u[1])

    # plt.figure(figsize=(25, 3))
    # plt.xlim([0, avg_length*len(l_uc)])
    # plt.ylim([0, 100])
    # plt.plot(uc)
    # plt.show()
    # plt.close()

    # # print(l)
    # # avg_length = np.array(length_list).mean()
    # # print(int(avg_length))

    ''' interpolation (List --> DataFrame --> Interpolate --> List again) '''
    # tmp_fhr = pd.DataFrame()
    # tmp_mhr = pd.DataFrame()
    # tmp_uc = pd.DataFrame()

    # tmp_fhr['fhr'] = fhr
    # tmp_mhr['mhr'] = mhr
    # tmp_uc['uc'] = uc

    # tmp_fhr = tmp_fhr.interpolate(method='linear')
    # tmp_mhr = tmp_mhr.interpolate(method='linear')
    # tmp_uc = tmp_uc.interpolate(method='linear')

    # fhr = list(tmp_fhr['fhr'])
    # mhr = list(tmp_mhr['mhr'])
    # uc = list(tmp_uc['uc'])
    ''' twin's MHR : 75 ????????? ????????? ??? : Twin red line ??? blue line ??? ???????????? MHR??? ????????? ????????? ???????????? ??????????????? ?????? ???????????????.'''

    for idx in range(len(mhr)):
        if mhr[idx] <= 75:
            pass
        else:
            mhr[idx] = np.nan

    data = pd.DataFrame(zip(fhr, mhr, uc), columns=['FHR', 'MHR', 'UC'])
    data.plot(figsize=(30, 3))
    plt.savefig(f'C:/Users/User5/Desktop/github/HR/result/waveplot_{p}.png', dpi=500)
    plt.show()


    data.to_csv(f'C:/Users/User5/Desktop/github/HR/result/raw_data_{p}.csv', index=False)



    total_pix = len(list(base_path.glob(f'{p}*.png'))) * (xr_new-xl_new)
    # waveform ?????? (????????????)
    total_sec = len(list(base_path.glob(f'{p}*.png'))) * 60 * 8
    nni = np.array([60/f for f in fhr if f is not None])
    # print('Total pix: ', total_pix)
    # print('Total sec: ', total_sec)
    # print("Hz: ", round(total_pix/total_sec))

    # print(nni)

    # FHR HRV
    result = hrv(nni=nni, sampling_rate=total_pix/total_sec, show=False)
    sdnn, sdann, rmssd, sdsd, nn50, pnn50, nn20, pnn20 = result['sdnn'], result['sdann'], result[
        'rmssd'], result['sdsd'], result['nn50'], result['pnn50'], result['nn20'], result['pnn20']
    vlf, lf, hf = result['lomb_abs'][0], result['lomb_abs'][1], result['lomb_abs'][2]
    lf_hf = lf/hf
    vlf_peak, lf_peak, hf_peak = result['lomb_peak'][0], result['lomb_peak'][1], result['lomb_peak'][2]
    vlf_rel, lf_rel, hf_rel = result['lomb_rel'][0], result['lomb_rel'][1], result['lomb_rel'][2]
    sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2 = result['sd1'], result['sd2'], result[
        'sd_ratio'], result['sampen'], result['dfa_alpha1'], result['dfa_alpha2']
    a_ratio = dfa_a1/dfa_a2
    # print(result)
    print(sdnn, sdann, rmssd, vlf_peak, vlf_rel, vlf, sd1, dfa_a1, a_ratio)

    if hrv_data is None:
        hrv_data = np.array([[p, sdnn, sdann, rmssd, sdsd, pnn50, pnn20, vlf, lf, hf, lf_hf, vlf_peak,
                              lf_peak, hf_peak, vlf_rel, lf_rel, hf_rel, sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2, a_ratio]])
    else:
        hrv_data = np.concatenate((hrv_data, np.array([[p, sdnn, sdann, rmssd, sdsd, pnn50, pnn20, vlf, lf, hf, lf_hf, vlf_peak,
                                                        lf_peak, hf_peak, vlf_rel, lf_rel, hf_rel, sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2, a_ratio]])))

    hrv_df = pd.DataFrame(hrv_data, columns=['pid', 'sdnn', 'sdann', 'rmssd', 'sdsd', 'pnn50', 'pnn20', 'vlf', 'lf', 'hf', 'lf/hf',
                                            'vlf_peak', 'lf_peak', 'hf_peak', 'vlf_rel', 'lf_rel', 'hf_rel', 'sd1', 'sd2', 'sd_ratio', 'sampen', 'dfa_a1', 'dfa_a2', 'a_ratio'])
    hrv_df.to_csv(f'C:/Users/User5/Desktop/github/HR/result/FHR_hrv_{p}.csv', index=False)


    # MHR HRV
    total_pix = len(list(base_path.glob(f'{p}*.png'))) * (xr_new-xl_new)
    # waveform ?????? (????????????)
    total_sec = len(list(base_path.glob(f'{p}*.png'))) * 60 * 8
    nni = np.array([60/f for f in mhr if f is not None])
    # pd.DataFrame(nni).to_csv('./test.csv')
    # print(len(nni))
    # print(nni)

    result = hrv(nni=nni, sampling_rate=total_pix/total_sec, show=False)
    sdnn, sdann, rmssd, sdsd, nn50, pnn50, nn20, pnn20 = result['sdnn'], result['sdann'], result[
        'rmssd'], result['sdsd'], result['nn50'], result['pnn50'], result['nn20'], result['pnn20']
    vlf, lf, hf = result['lomb_abs'][0], result['lomb_abs'][1], result['lomb_abs'][2]
    lf_hf = lf/hf
    vlf_peak, lf_peak, hf_peak = result['lomb_peak'][0], result['lomb_peak'][1], result['lomb_peak'][2]
    vlf_rel, lf_rel, hf_rel = result['lomb_rel'][0], result['lomb_rel'][1], result['lomb_rel'][2]
    sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2 = result['sd1'], result['sd2'], result[
        'sd_ratio'], result['sampen'], result['dfa_alpha1'], result['dfa_alpha2']
    a_ratio = dfa_a1/dfa_a2
    # print(result)
    print(sdnn, sdann, rmssd, vlf_peak, vlf_rel, vlf, sd1, dfa_a1, a_ratio)

    if hrv_data2 is None:
        hrv_data2 = np.array([[p, sdnn, sdann, rmssd, sdsd, pnn50, pnn20, vlf, lf, hf, lf_hf, vlf_peak,
                                lf_peak, hf_peak, vlf_rel, lf_rel, hf_rel, sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2, a_ratio]])
    else:
        hrv_data2 = np.concatenate((hrv_data2, np.array([[p, sdnn, sdann, rmssd, sdsd, pnn50, pnn20, vlf, lf, hf, lf_hf, vlf_peak,
                                                        lf_peak, hf_peak, vlf_rel, lf_rel, hf_rel, sd1, sd2, sd_ratio, sampen, dfa_a1, dfa_a2, a_ratio]])))

    hrv_df2 = pd.DataFrame(hrv_data2, columns=['pid', 'sdnn', 'sdann', 'rmssd', 'sdsd', 'pnn50', 'pnn20', 'vlf', 'lf', 'hf', 'lf/hf',
                                            'vlf_peak', 'lf_peak', 'hf_peak', 'vlf_rel', 'lf_rel', 'hf_rel', 'sd1', 'sd2', 'sd_ratio', 'sampen', 'dfa_a1', 'dfa_a2', 'a_ratio'])
    hrv_df2.to_csv(f'C:/Users/User5/Desktop/github/HR/result/MHR_hrv_{p}.csv', index=False)

