
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")

from __future__ import print_function
from __future__ import division
import pandas
import numpy as np
import matplotlib.pyplot as plt
from ethomap.tsne import ParametricTSNE

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max


# In[161]:



#Import the csv data from the file 
file_names = ['20170513_Kali-3381.csv',
              '20170514_Lio-3383.csv',
              '20170515_Elly-3383.csv',
              '20170518_Sammy-4084.csv',
              '20170519_Lissy-3381.csv',
              '20170522_Lucy-3381.csv',
              '20170523_Dudek-4084.csv',
              '20170525_Flecky-4087.csv',
              '20170530_Lasse-4087.csv',
              '201710513_Baghira-4084.csv']
df = []
df_valid_position_loc = []
df_clean = []
positions = []
for name_indx, name in enumerate(file_names):
    temp_df = pandas.read_csv('Data/' + name)
    #dealing with human walking data at the beginning
    if name_indx == 6:
        temp_df = temp_df[1666:]
    df.append(temp_df)
    df_valid_position_loc.append(
        temp_df[['location-long', 'location-lat']].notnull())
#df.describe() #this gives a summary of the dataframe columns

for df_indx in range(len(df)):
    df_clean.append(df[df_indx][df_valid_position_loc[df_indx].all(axis=1)])
    positions.append(df_clean[df_indx].as_matrix(['location-long', 'location-lat']))
print(positions[0][-1,1])


# In[162]:

figure = 6
plt.plot(np.sort(positions[figure][:,0]))
plt.figure()
print(np.sort(positions[figure][:,0]))
print(positions[figure][:,1].max())
plt.plot(positions[figure][:,0], positions[figure][:,1])


# In[163]:

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)


# In[165]:

length_of_segment = 21
half_of_segment = 10


def rotate_2d(matrix, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    rotated_matrix = np.matmul(rotation_matrix, matrix.transpose()).transpose()
    return rotated_matrix

def get_segments(loc, half_seg_length, scaling):
    num_points = 0
    for indx in range(len(loc)):
        num_points += (loc[indx].shape[0] - 2 * half_seg_length)
        
    segs = np.zeros((num_points, 2*half_seg_length+1, 2))
    counter = 0
    
    for indx, df_local in enumerate(loc):
        for seg_indx in range(df_local.shape[0] - (2 * half_seg_length)-1):
            segment = np.copy(df_local[seg_indx:(seg_indx + 2*half_seg_length + 1), :]) #Why do I need the copy?
            segment -= df_local[seg_indx + half_seg_length, :]        
            segs[counter + seg_indx, :, :] = segment 
            segs[counter + seg_indx, :, :] *= scaling
            
        counter += df_local.shape[0] - 2 * half_seg_length

    return segs

def get_vector_magnitude(segments):
    distances = np.ones((segments.shape[0], segments.shape[1]))
    for seg_indx in range(segments.shape[0]):
            distances[seg_indx] = np.sqrt(segments[seg_indx, :, 0] ** 2 +
                                               segments[seg_indx, :, 1] ** 2)
    return(distances)

def get_vector_from_last(segments):
    distances = np.zeros_like(segments)
    for indx in range(0, segments.shape[1]-1):
        distances[:, indx] = segments[:, indx + 1,:] - segments[:, indx,:]
    return distances

def get_mean_distance_from_local_origin(segments):
    distances = get_vector_magnitude(segments)
    distances = np.sum(distances, 1)
    distances /= segments.shape[1]
    return(distances)

def get_mean_step_displacement(segments):
    mean_displacement = np.zeros(segments.shape[0])
    displacement = get_vector_from_last(segments)
    displacement = get_vector_magnitude(displacement)
    mean_displacement = np.sum(displacement, 1)
    mean_displacement /= segments.shape[1]
    return mean_displacement

def get_mean_net_displacement(segments):
    distances = np.zeros(segments.shape[0])
    difference_vector = segments[:, -1] - segments[:, 0]
    distances = np.sqrt(difference_vector[:,0] ** 2 + difference_vector[:,1] ** 2)
    distances /= segments.shape[1]
    return(distances)


    
            
            

path_segments = get_segments(positions, half_of_segment, 10000.0)

print('path_segments.shape: ', path_segments.shape)
print('min: ', path_segments.min(), '\n', 'max: ', path_segments.max())

mean_net_displacement = get_mean_net_displacement(path_segments)
mean_step_displacement = get_mean_step_displacement(path_segments)
mean_distance_from_local_origin = get_mean_distance_from_local_origin(path_segments)

tsne_input = np.column_stack((mean_net_displacement, mean_step_displacement, mean_distance_from_local_origin))

print('tsne_input: ', tsne_input.shape)
print('min: ', tsne_input[:,0].min(), '\n', 'max: ', tsne_input[:,0].max())
print('min: ', tsne_input[:,1].min(), '\n', 'max: ', tsne_input[:,1].max())
print('min: ', tsne_input[:,2].min(), '\n', 'max: ', tsne_input[:,2].max())


# In[169]:

#check distributions of the tSNE input parameters
min_range = 15
max_range = -15
num_bins = 1000
print('mean_net_displacement')
plt.figure()
plt.hist(mean_net_displacement[min_range:max_range], num_bins)
plt.show()
print('mean_step_displacement')
plt.figure()
plt.hist(np.sort(mean_step_displacement[min_range:max_range]), num_bins)
plt.show()
print('mean_distance_from_local_origin')
plt.figure()
plt.hist(np.sort(mean_distance_from_local_origin[min_range:max_range]), num_bins)
plt.show()


# In[174]:

#View some demo segments
sampling = 1000
for plt_indx in range(2):
    plt.figure()
    plt.plot(path_segments[plt_indx * sampling,:,0], path_segments[plt_indx * sampling,:,1])


# In[6]:

#Train the ptSNE embedding
tsne_distance = ParametricTSNE(n_jobs=-1, metric='cosine', verbose=1)
#use every other point for training
tsne_input_sample = tsne_input[::2] 
print('using ' , len(tsne_input_sample), ' points for training.')
tsne_distance.fit(tsne_input_sample)


# In[7]:

#Embed all the points in tSNE space
result = tsne_distance.transform(tsne_input)


# In[8]:

#create dataframe
#Check they are the same size
print(len(tsne_input))
print(len(result))

names = []
#Don't include the points at the beginning and end that didn't have enough padding to be classified
pos = positions[0][half_of_segment:-half_of_segment]
for indx in range(1, len(positions)):
    pos = np.vstack((pos, positions[indx][half_of_segment:-half_of_segment]))

#Prepare a name column for the data frame
for indx in range(0, len(positions)):
    for __ in range(len(positions[indx][half_of_segment:-half_of_segment])):
        names.append(file_names[indx])
        
names = np.asarray(names)
names = np.expand_dims(names, 1)
print(names.shape)
data = np.hstack((names,pos, tsne_input, result)) 

df_out = pandas.DataFrame(data, columns=['file',
                                         'long', 
                                         'lat', 
                                         'mean_net_displacement', 
                                         'mean_step_displacement', 
                                         'mean_distance_from_local_origin', 
                                         'tsne_dim1', 
                                         'tsne_dim2'
                                        ])


# In[9]:

df_out


# In[10]:

df_out.to_csv('behavioral_quantization_3.csv')


# In[ ]:




# In[3]:

import ethomap
import pickle


# In[4]:

imported_df = pandas.read_csv('behavioral_quantization_3.csv')
imported_df.describe()


# In[5]:

tsne = imported_df.as_matrix(['tsne_dim1', 'tsne_dim2'])
tsne


# In[6]:

size_of_behavior_space = 100
Y, density = ethomap.mapping.point_density(tsne, 1.0, size_of_behavior_space)
#density = np.where(density>.00000000000001, density, np.zeros_like(density))


# In[7]:

# Generate the markers as local maxima of the distance to the background
colormap = plt.cm.spectral 

local_maxi = peak_local_max(density, indices=False, min_distance = 13)
markers = ndi.label(local_maxi)[0]
labels = watershed(-density, markers, mask=density)
number_of_states = labels.max()

fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(density, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('t-SNE space point density')
ax[1].imshow(labels, cmap=colormap, interpolation='nearest')
ax[1].set_title('Behavioral segmentation')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

print('There are ', number_of_states, ' behavioral types.')


# In[8]:

plt.figure(figsize=(3, 3), dpi=100)
plt.imshow((density), cmap='viridis')
plt.axis('off')
plt.show()


# In[9]:

plt.figure(figsize=(3, 3), dpi=100)
plt.imshow(labels, cmap=colormap, interpolation='nearest')
plt.axis('off')
plt.show()


# In[13]:

#Calculate each segments segmentation label
segment_start = 0
segment_end = None
tsne_d1 = np.squeeze(np.asmatrix(imported_df['tsne_dim1'][segment_start:segment_end]))
tsne_d2 = np.squeeze(np.asmatrix(imported_df['tsne_dim2'][segment_start:segment_end]))
tsne_d1_scaled = size_of_behavior_space * (tsne_d1 - tsne_d1.min()) / (tsne_d1.max() - tsne_d1.min())
tsne_d2_scaled = size_of_behavior_space * (tsne_d2 - tsne_d2.min()) / (tsne_d2.max() - tsne_d2.min())
tsne_d1_scaled = tsne_d1_scaled.astype(int)
tsne_d2_scaled = tsne_d2_scaled.astype(int)
color = labels[tsne_d1_scaled, tsne_d2_scaled] 

imported_df['behavioral_label'] = color[0]
#imported_df = imported_df.drop('Unnamed: 0', axis=1)

#save dataframe to .csv
imported_df.to_csv('cats_labeled_behavioral_quantization.csv')


# In[151]:

imported_df


# In[14]:

#plot all of the trajectories
x = imported_df['long'][segment_start:segment_end]
y = imported_df['lat'][segment_start:segment_end] 
fig = plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(x, y, s = 2, c = imported_df['behavioral_label'], cmap=colormap)
plt.xlim(imported_df['long'][segment_start:segment_end].min(), imported_df['long'][segment_start:segment_end].max())
plt.ylim(imported_df['lat'][segment_start:segment_end].min(), imported_df['lat'][segment_start:segment_end].max())



# In[15]:

country_cats = [
              '20170515_Elly-3383.csv',
              '20170518_Sammy-4084.csv',
              '20170519_Lissy-3381.csv',
              '20170522_Lucy-3381.csv',
              '20170523_Dudek-4084.csv',
              '20170530_Lasse-4087.csv',]

city_cats = ['20170513_Kali-3381.csv',
              '20170514_Lio-3383.csv',
              '20170525_Flecky-4087.csv',
              '201710513_Baghira-4084.csv']

all_cats = country_cats + city_cats

city_df = imported_df[imported_df['file'].isin(city_cats)]
country_df = imported_df[imported_df['file'].isin(country_cats)]


# In[16]:

environment =  all_cats
environment_df = imported_df[imported_df['file'].isin(environment)]


# In[17]:

plt.figure()
fig = plt.figure(figsize=(3, 3), dpi=100)
fig.suptitle('Town cat movement behavior distribution', fontsize=16, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_xlabel('Behavioral type', fontsize=12)
ax.set_ylabel('Number of segments', fontsize=12)
n, bins, patches = plt.hist(city_df['behavioral_label'], bins=number_of_states)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
cm = colormap
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.show()

fig = plt.figure(figsize=(3, 3), dpi=100)
fig.suptitle('Rural cat movement behavior distribution', fontsize=16, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_xlabel('Behavioral type', fontsize=12)
ax.set_ylabel('Number of segments', fontsize=12)

n, bins, patches = plt.hist(country_df['behavioral_label'], bins=number_of_states)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
cm = colormap

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))


# In[18]:

size_graph = 5
for cat_indx in range(len(environment)):
    single_cat = environment_df[environment_df['file'] == environment[cat_indx]] 
    scale_long = single_cat['long'][segment_start:segment_end].max() - single_cat['long'][segment_start:segment_end].min() 
    scale_lat = single_cat['lat'][segment_start:segment_end].max() - single_cat['lat'][segment_start:segment_end].min()
    scale = scale_lat / scale_long 
    fig = plt.figure(figsize=(size_graph, size_graph * scale), dpi=100)
    fig.suptitle((np.asmatrix(single_cat['file'])[0,0] + ' with points labeled with t-SNE segmentation'),
                 fontsize=10, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    plt.scatter(single_cat['long'], single_cat['lat'], c=single_cat['behavioral_label'], s=2.0, cmap=colormap)
    plt.xlim(single_cat['long'][segment_start:segment_end].min(), single_cat['long'][segment_start:segment_end].max())
    plt.ylim(single_cat['lat'][segment_start:segment_end].min(), single_cat['lat'][segment_start:segment_end].max())
    fig = plt.figure()
    plt.show()



# In[20]:

environment = all_cats
for cat_indx in range(len(environment)):
    single_cat = environment_df[environment_df['file'] == environment[cat_indx]]
    fig = plt.figure(figsize=(3, 3), dpi=100)
    fig.suptitle((np.asmatrix(single_cat['file'])[0,0] + ' movement behavior distribution'),
                 fontsize=10, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Behavioral type', fontsize=12)
    ax.set_ylabel('Number of segments', fontsize=12)
    n, bins, patches = plt.hist(single_cat['behavioral_label'], bins=number_of_states)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cm = colormap

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.show()


# In[ ]:



