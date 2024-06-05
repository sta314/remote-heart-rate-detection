# %%
import os
import cv2
import neurokit2 as nk
import numpy as np
import dlib
import matplotlib.pyplot as plt

# %%
root = './UBFC_DATASET/DATASET_1/'

dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

for dir_name in dirs:
    print("Starting the execution for the video:", dir_name)

    vid_folder = os.path.join(root, dir_name)

    gt_filename = os.path.join(vid_folder, 'gtdump.xmp')
    if os.path.isfile(gt_filename):
        gt_data = np.genfromtxt(gt_filename, delimiter=',')
        gt_time = gt_data[:, 0] / 1000
        gt_hr = gt_data[:, 1]
        gt_trace = gt_data[:, 3]
    else:
        gt_filename = os.path.join(vid_folder, 'ground_truth.txt')
        if os.path.isfile(gt_filename):
            gt_data = np.loadtxt(gt_filename,)
            gt_trace = gt_data[0,:]
            gt_hr = gt_data[1,:]
            gt_time = gt_data[2,:]


    gt_trace = (gt_trace - np.mean(gt_trace)) / np.std(gt_trace)

    # %%
    print("dir_name:", dir_name)
    print("vid_folder:", vid_folder)
    print("gt_trace shape:", gt_trace.shape)
    print("gt_hr shape:", gt_hr.shape)
    print("gt_time shape:", gt_time.shape)

    # %%
    import dlib
    import cv2
    vid_path = os.path.join(vid_folder, 'vid.avi')
    vidObj = cv2.VideoCapture(vid_path)
    face_list = []

    fps = vidObj.get(cv2.CAP_PROP_FPS)
    first_frame = True
    tracker = dlib.correlation_tracker()

    # Load the face detector and the shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

    left_cheek_signal = []
    right_cheek_signal = []
    forehead_signal = []
    chin_signal = []

    # Function to extract facial landmarks
    def extract_facial_landmarks(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        landmarks = []
        for face in faces:
            shape = predictor(gray, face)
            for i in range(81):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))
        return landmarks

    def calculate_mean_pixel_values(frame, points):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        mean_values = np.sum(masked_frame, axis=(0, 1)) / np.count_nonzero(mask)
        return mean_values

    # Open the video file
    vidObj = cv2.VideoCapture(vid_path)

    # Process each frame of the video
    while True:
        ret, frame = vidObj.read()
        if not ret:
            break
        
        # Extract facial landmarks
        landmarks = extract_facial_landmarks(frame)

        # Extract signals from corresponding face patches
        left_cheek_signal.append(calculate_mean_pixel_values(frame, np.array(landmarks)[[4, 29, 1]]))
        right_cheek_signal.append(calculate_mean_pixel_values(frame, np.array(landmarks)[[29, 12, 15]]))
        forehead_signal.append(calculate_mean_pixel_values(frame, np.array(landmarks)[[19, 24, 72, 69]]))
        chin_signal.append(calculate_mean_pixel_values(frame, np.array(landmarks)[[8, 12, 57, 4]]))
        

    left_cheek_signal    = np.array(left_cheek_signal)
    right_cheek_signal   = np.array(right_cheek_signal)
    forehead_signal      = np.array(forehead_signal)
    chin_signal          = np.array(chin_signal)

    # %%
    # Calculating means

    means = np.array([left_cheek_signal, right_cheek_signal, forehead_signal, chin_signal])
    for i in range(means.shape[0]):
        means[i, :,0] = (means[i, :,0] - np.average(means[i, :,0]))/np.std(means[i, :,0])
        means[i, :,1] = (means[i, :,1] - np.average(means[i, :,1]))/np.std(means[i, :,1])
        means[i, :,2] = (means[i, :,2] - np.average(means[i, :,2]))/np.std(means[i, :,2])

    # Detrending means

    detrended_means = means
    for i in range(means.shape[0]):
        detrended_means[i, :,0] = nk.signal_detrend(means[i, :,0], method="tarvainen2002", regularization=120)
        detrended_means[i, :,1] = nk.signal_detrend(means[i, :,1], method="tarvainen2002", regularization=120)
        detrended_means[i, :,2] = nk.signal_detrend(means[i, :,2], method="tarvainen2002", regularization=120)

    # Moving average filter

    window_size = int(fps*0.18)
    kernel = np.ones(window_size) / window_size
    filtered_means = detrended_means

    for i in range(means.shape[0]):
        filtered_means[i, :,0] = np.convolve(filtered_means[i, :,0],kernel,'same')
        filtered_means[i, :,1] = np.convolve(filtered_means[i, :,1],kernel,'same')
        filtered_means[i, :,2] = np.convolve(filtered_means[i, :,2],kernel,'same')


    # FastICA decomposition

    from sklearn.decomposition import FastICA

    transformer = FastICA(n_components = 3)

    independent_signals = np.zeros((means.shape))
    for i in range(means.shape[0]):
        independent_signals[i] = transformer.fit_transform(filtered_means[i])

    # Calculating PSD

    from scipy.signal import periodogram

    signals = np.zeros(independent_signals.shape[:2])
    for i in range(means.shape[0]):
        (f_r, S_r) = periodogram(independent_signals[i, :,0], fps, scaling='density')
        (f_g, S_g) = periodogram(independent_signals[i, :,1], fps, scaling='density')
        (f_b, S_b) = periodogram(independent_signals[i, :,2], fps, scaling='density')

        signals[i] = independent_signals[i, :,np.argmax([max(S_r),max(S_g),max(S_b)])]

    # Applying bandpass filter between 0.8 Hz and 2 Hz

    import numpy as np
    from scipy.signal import butter, lfilter

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    final_signals = np.zeros(signals.shape[:2])
    for i in range(signals.shape[0]):
        final_signals[i] = butter_bandpass_filter(signals[i], 0.8, 2.2, fps, order=5)

    # Selecting the best ROI in terms of PSD calculations

    idx_max = -1
    S_max = -np.inf

    for i in range(means.shape[0]):
        f, S = periodogram(final_signals[i], fps, scaling='density')
        if max(S) > S_max:
            idx_max = i
            S_max = max(S)

    final_signal = final_signals[idx_max]

    # %%
    # Last calculations for 6 seconds window (3s from each side)

    # Calculating heart rate values
    values = []

    window = 3

    lower_bound = window
    upper_bound = int(final_signal.shape[0] / fps) - window

    for i in range(lower_bound, upper_bound, 1):
        x = (final_signal[int((i - window) * fps):int((i + window) * fps)])
        (f_r, S_r) = periodogram(x, fps, nfft=len(x) * 4, scaling='density')

        # Taking frequency with the highest power as the pulse signal
        f = f_r[S_r.argmax()]

        hr = 60 *f
        if i == window:
            values.extend([hr] * window)

        values.append(hr)

    values.extend([values[-1]] * window)

    # Saving the comparison plot

    results_folder = f"results/{dir_name}_{window}/"
    os.makedirs(results_folder, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    plt.figure(figsize=(8, 4))
    plt.plot(values, label='Estimation')
    plt.plot(gt_time, gt_hr, label='Ground Truth')
    plt.xlabel('Seconds')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_folder + 'comparison_plot.png', dpi=300)
    plt.clf()

    # Saving the predictions, just in case

    np.save(results_folder + 'predictions.npy', np.array(values))

    # Saving the mean hr, mean predictions and difference between them

    mean_hr = np.mean(gt_hr)
    mean_predictions = np.mean(values)
    difference = np.abs(mean_hr - mean_predictions)

    np.savetxt(results_folder + 'wholemeans.txt', [mean_hr, mean_predictions, difference], fmt='%.6f')

    # Saving the similar information but difference is calculated per seconds

    # window gt_hr in windows with length fps and print the mean of each window
    gt_hr_windows = []
    step_size = int(gt_hr.shape[0] / len(values))
    for i in range(0, gt_hr.shape[0], step_size):
        gt_hr_windows.append(np.mean(gt_hr[i :i + step_size]))
    gt_hr_windows = gt_hr_windows[:len(values)]


    values = np.array(values)
    gt_hr_windows = np.array(gt_hr_windows)

    # Calculate the MAE and MSE between gt_hr_windows and values

    MAE = np.mean(np.abs(gt_hr_windows - values))
    RMSE = np.sqrt(np.mean(np.square(gt_hr_windows - values)))
    MAPE = np.mean(np.abs((gt_hr_windows - values) / gt_hr_windows)) * 100
    PEACORR, _ = pearsonr(gt_hr_windows, values)

    np.savetxt(results_folder + 'persecondmeans.txt', [np.mean(gt_hr_windows), np.mean(values), MAE, RMSE, MAPE, PEACORR], fmt='%.6f')

    # %%
    # Last calculations for 8 seconds window (4s from each side)

    # Calculating heart rate values
    values = []

    window = 4

    lower_bound = window
    upper_bound = int(final_signal.shape[0] / fps) - window

    for i in range(lower_bound, upper_bound, 1):
        x = (final_signal[int((i - window) * fps):int((i + window) * fps)])
        (f_r, S_r) = periodogram(x, fps, nfft=len(x) * 4, scaling='density')

        # Taking frequency with the highest power as the pulse signal
        f = f_r[S_r.argmax()]

        hr = 60 *f
        if i == window:
            values.extend([hr] * window)

        values.append(hr)

    values.extend([values[-1]] * window)

    # Saving the comparison plot

    results_folder = f"results/{dir_name}_{window}/"
    os.makedirs(results_folder, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    plt.figure(figsize=(8, 4))
    plt.plot(values, label='Estimation')
    plt.plot(gt_time, gt_hr, label='Ground Truth')
    plt.xlabel('Seconds')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_folder + 'comparison_plot.png', dpi=300)
    plt.clf()

    # Saving the predictions, just in case

    np.save(results_folder + 'predictions.npy', np.array(values))

    # Saving the mean hr, mean predictions and difference between them

    mean_hr = np.mean(gt_hr)
    mean_predictions = np.mean(values)
    difference = np.abs(mean_hr - mean_predictions)

    np.savetxt(results_folder + 'wholemeans.txt', [mean_hr, mean_predictions, difference], fmt='%.6f')

    # Saving the similar information but difference is calculated per seconds

    # window gt_hr in windows with length fps and print the mean of each window
    gt_hr_windows = []
    step_size = int(gt_hr.shape[0] / len(values))
    for i in range(0, gt_hr.shape[0], step_size):
        gt_hr_windows.append(np.mean(gt_hr[i :i + step_size]))
    gt_hr_windows = gt_hr_windows[:len(values)]


    values = np.array(values)
    gt_hr_windows = np.array(gt_hr_windows)

    # Calculate the MAE and MSE between gt_hr_windows and values

    MAE = np.mean(np.abs(gt_hr_windows - values))
    RMSE = np.sqrt(np.mean(np.square(gt_hr_windows - values)))
    MAPE = np.mean(np.abs((gt_hr_windows - values) / gt_hr_windows)) * 100
    PEACORR, _ = pearsonr(gt_hr_windows, values)

    np.savetxt(results_folder + 'persecondmeans.txt', [np.mean(gt_hr_windows), np.mean(values), MAE, RMSE, MAPE, PEACORR], fmt='%.6f')

    # %%
    # Last calculations for 12 seconds window (6s from each side)

    # Calculating heart rate values
    values = []

    window = 6

    lower_bound = window
    upper_bound = int(final_signal.shape[0] / fps) - window

    for i in range(lower_bound, upper_bound, 1):
        x = (final_signal[int((i - window) * fps):int((i + window) * fps)])
        (f_r, S_r) = periodogram(x, fps, nfft=len(x) * 4, scaling='density')

        # Taking frequency with the highest power as the pulse signal
        f = f_r[S_r.argmax()]

        hr = 60 *f
        if i == window:
            values.extend([hr] * window)

        values.append(hr)

    values.extend([values[-1]] * window)

    # Saving the comparison plot

    results_folder = f"results/{dir_name}_{window}/"
    os.makedirs(results_folder, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    plt.figure(figsize=(8, 4))
    plt.plot(values, label='Estimation')
    plt.plot(gt_time, gt_hr, label='Ground Truth')
    plt.xlabel('Seconds')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_folder + 'comparison_plot.png', dpi=300)
    plt.clf()

    # Saving the predictions, just in case

    np.save(results_folder + 'predictions.npy', np.array(values))

    # Saving the mean hr, mean predictions and difference between them

    mean_hr = np.mean(gt_hr)
    mean_predictions = np.mean(values)
    difference = np.abs(mean_hr - mean_predictions)

    np.savetxt(results_folder + 'wholemeans.txt', [mean_hr, mean_predictions, difference], fmt='%.6f')

    # Saving the similar information but difference is calculated per seconds

    # window gt_hr in windows with length fps and print the mean of each window
    gt_hr_windows = []
    step_size = int(gt_hr.shape[0] / len(values))
    for i in range(0, gt_hr.shape[0], step_size):
        gt_hr_windows.append(np.mean(gt_hr[i :i + step_size]))
    gt_hr_windows = gt_hr_windows[:len(values)]


    values = np.array(values)
    gt_hr_windows = np.array(gt_hr_windows)

    # Calculate the MAE and MSE between gt_hr_windows and values

    MAE = np.mean(np.abs(gt_hr_windows - values))
    RMSE = np.sqrt(np.mean(np.square(gt_hr_windows - values)))
    MAPE = np.mean(np.abs((gt_hr_windows - values) / gt_hr_windows)) * 100
    PEACORR, _ = pearsonr(gt_hr_windows, values)

    np.savetxt(results_folder + 'persecondmeans.txt', [np.mean(gt_hr_windows), np.mean(values), MAE, RMSE, MAPE, PEACORR], fmt='%.6f')


