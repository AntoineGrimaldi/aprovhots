from numpy import genfromtxt, save

csv_path = '/home/amelie/Scripts/Data/coastline_events/DVS128_ZED_NUC_jAER/session_6_only_sea/DVS128_06_06_2022_session_6.csv';
events = genfromtxt(csv_path, delimiter=',')
print(events)
save(csv_path.replace('.csv', '.npy'), events)
print("Events correctly saved as", csv_path.replace('.csv', '.npy'))