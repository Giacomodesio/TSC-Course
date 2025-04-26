"""
Created on 01 March 2025

@author: De Sio Giacomo

@ Title: SVD applied to the plasma 16 kV no Humidity
"""

import numpy as np
import matplotlib.pyplot as plt
import time # Usefull for timing the code
import pandas as pd
import os 

# Prepare the folder with the unzipped data
FOLDER=r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\EveryFrame"

# For info, these data were collected with the following parameters
n_t=500 # number of steps.
Fs=5 # Sampling frequency
dt=1/Fs # This data was sampled at 2kHz.
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'PIVlab_%04d'%5+'.txt' # Check it out: print(Name)
# Read data from a file
# DATA = pd.read_csv(Name, delimiter=',') # Here we have the four colums
Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
nxny=Dat.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s=2*nxny
## 1. Reconstruct Mesh from file
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
# Number of n_X/n_Y from forward differences
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
# Reshaping the grid from the data
n_x=(nxny//(n_y)) # Carefull with integer and float!
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y))) # This is now the mesh! 60x114.

####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################

for k in range(1,n_t+1):
  Name=FOLDER+os.sep+'PIVlab_%04d'%k+'.txt' # Name of the file to read
  # Read data from a file
  Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
  V_X=np.array(Dat['u [m/s]']) # U component
  V_Y=np.array(Dat['v [m/s]']) # V component
  D[:,k-1]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k)+'/'+str(n_t)) 

  
# For a stationary test case like this, you might want to remove the mean  
D_MEAN=np.mean(D,1) # Temporal average (along the columns)
D_Mr=D-np.array([D_MEAN,]*n_t).transpose() # Mean Removed

# Check the mean flow
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
# Put both components as fields in the grid
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
# Or you can plot it as streamlines
contour = plt.contourf(Xg,Yg,Magn, cmap='viridis')
cbar = plt.colorbar(contour)

# One possibility is to use quiver
STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 16 kV - No humidity',fontsize=18)


####################### 2. CONSTRUCT THE BASIS PSI_P #################
# In the POD, the temporal basis comes as eigenvectors of K:
# First we compute K
print('Computing Correlation Matrix')
K = np.dot(D_Mr.transpose(), D_Mr) # Comput temporal correlation matrix, DDFM course --> lecture 5 --> slide 8 
# Comput the Temporal basis Psi
Psi_P, Lambda_P, _ = np.linalg.svd(K)


####################### 3. COMPUTING THE SPATIAL STRUCTURES #################

Sigma_P=Lambda_P**0.5 # Have a look at how strong these decay !!!!

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
plt.plot(Sigma_P,'ko:')
plt.xscale('linear')
plt.yscale('linear')

# We take only the first 200 modes.
Sigma_P_t=Sigma_P[0:200]
Sigma_P_Inv_V=1/Sigma_P_t
# Accordingly we reduce psi_P
Psi_P_t=Psi_P[:,0:200]
# So we have the inverse
Sigma_P_Inv=np.diag(Sigma_P_Inv_V)


# We put some messages
print('Projecting Data')
Begin=time.time()
# This is Phi_P = D Psi Sigma_inv
Phi_P=np.linalg.multi_dot([D,Psi_P[:,0:200],Sigma_P_Inv])
Duration=time.time()-Begin # Measure the time of the projection
print('Decomposition completed in  '+str(Duration)+' seconds')
# Check the decomposition convergence (carefull: we have only 200 modes)
D_P=np.linalg.multi_dot([Phi_P,np.diag(Sigma_P_t),np.transpose(Psi_P_t)]) 
Error=np.linalg.norm(D-D_P)/np.linalg.norm(D)
print('Convergence Error: E_C='+"{:.2f}".format(Error/100)+' %')

"""
Energy

"""
Energy = np.zeros((len(Sigma_P),1))
Energy = Sigma_P**2/np.sum(Sigma_P**2)
cumulative = np.cumsum(Energy)/np.sum(Energy)

X_Axis = np.arange(Energy.shape[0])


fig, axes = plt.subplots(1, 2, figsize = (12,4))
ax = axes[0]
ax.bar(X_Axis, Energy, width=0.5)
ax.set_xlim(-0.25,30)
ax.set_xlabel('Modes')
ax.set_ylabel('Energy Content')

ax = axes[1]
ax.plot(cumulative, marker = 'o', markerfacecolor = 'none', markeredgecolor = 'k', ls='-', color = 'k')
ax.set_xlabel('Modes')
ax.set_ylabel('Cumulative Energy')
ax.set_xlim(0, 300)

plt.show()

# We now export the Phi, the Psi and the Psi_HAT
# So we look at the Freqs vector
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins


# Export the first r modes
Fol_Out= r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\PODmodes"
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
  
for r in range(0,6):
  print('Exporting Mode '+str(r))  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]

  # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
  # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Show The spatial structure with the spectra of the associated temporal
  # structure

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})  # 2:1 rapporto di altezza

  # First subplot
  c = ax1.contourf(Xg, Yg, Magn, cmap = 'viridis')
  cbar = plt.colorbar(c)
  STEPx, STEPy = 2, 2
  ax1.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k', scale=0.8)
    
  ax1.set_aspect('equal')
  ax1.set_xlabel('$x[mm]$', fontsize=14)
  ax1.set_ylabel('$y[mm]$', fontsize=14)
  ax1.set_title(f"$\\phi_{r}(x,y)$", fontsize=18)
    
  # Second subplot: FFT
  ax2.plot(Freqs, np.abs(np.fft.fft(Psi)), 'ko')
  ax2.set_xlabel('$f[Hz]$', fontsize=16)
  ax2.set_ylabel(r'$\hat{\psi}_\mathcal{F}$', fontsize=16)
  ax2.set_title(rf'$\hat{{\psi}}_{{\mathcal{{F}}{r}}}$', fontsize=18)
  ax2.set_xlim([-3, 3])
    
  # Aggiusta il layout per evitare sovrapposizioni
  plt.subplots_adjust(hspace=0.3)  # Modifica lo spazio verticale tra i subplot
    
  # Salvataggio
  #plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.pdf', dpi=100)
  #plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.png', dpi=100)
    
  plt.show()

"""
Phase portrait Mode 0-1 --> Example of Characteristic pattern 
"""
mode_1_temporal = Psi_P[:,0]
mode_2_temporal = Psi_P[:,1]

plt.figure(figsize=(6, 6))  
plt.scatter(mode_1_temporal, mode_2_temporal, edgecolors='black', facecolors='none', s=20)  

plt.title('Phase portrait (0-1)', fontsize=16)
plt.xlabel('Mode 0', fontsize=14)
plt.ylabel('Mode 1', fontsize=14)

# Personalizza l'aspetto degli assi
plt.grid(True)  # Aggiungi la griglia
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostra il grafico
plt.tight_layout()  # Ottimizza il layout
plt.show()

"""
Phase portrait Modes 1-3: Example of No characteristic pattern
"""
mode_1_temporal = Psi_P[:,1]
mode_3_temporal = Psi_P[:,3]

plt.figure(figsize=(6, 6))  
plt.scatter(mode_1_temporal, mode_3_temporal, edgecolors='black', facecolors='none', s=20)  

plt.title('Phase portrait (1-3)', fontsize=16)
plt.xlabel('Mode 1', fontsize=14)
plt.ylabel('Mode 3', fontsize=14)

# Personalizza l'aspetto degli assi
plt.grid(True)  # Aggiungi la griglia
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostra il grafico
plt.tight_layout()  # Ottimizza il layout
plt.show()




#%%
import numpy as np
import matplotlib.pyplot as plt
import time # Usefull for timing the code
import pandas as pd
import os 
"""

Same thing as before but applied to the 17 kV case

"""
# Prepare the folder with the unzipped data
FOLDER=r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_17kV\EveryFrame"

# For info, these data were collected with the following parameters
n_t=500 # number of steps.
Fs=5 # Sampling frequency
dt=1/Fs # This data was sampled at 2kHz.
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'PIVlab_%04d'%5+'.txt' # Check it out: print(Name)
# Read data from a file
# DATA = pd.read_csv(Name, delimiter=',') # Here we have the four colums
Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
nxny=Dat.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s=2*nxny
## 1. Reconstruct Mesh from file
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
# Number of n_X/n_Y from forward differences
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
# Reshaping the grid from the data
n_x=(nxny//(n_y)) # Carefull with integer and float!
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y)))

####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################

for k in range(1,n_t+1):
  Name=FOLDER+os.sep+'PIVlab_%04d'%k+'.txt' # Name of the file to read
  # Read data from a file
  Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
  V_X=np.array(Dat['u [m/s]']) # U component
  V_Y=np.array(Dat['v [m/s]']) # V component
  D[:,k-1]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k)+'/'+str(n_t)) 

  
# For a stationary test case like this, you might want to remove the mean  
D_MEAN=np.mean(D,1) # Temporal average (along the columns)
D_Mr=D-np.array([D_MEAN,]*n_t).transpose() # Mean Removed

# Check the mean flow
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
# Put both components as fields in the grid
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
# Or you can plot it as streamlines
contour = plt.contourf(Xg,Yg,Magn, cmap='viridis')
cbar = plt.colorbar(contour)

# One possibility is to use quiver
STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 17 kV - No Humidity',fontsize=18)

####################### 2. CONSTRUCT THE BASIS PSI_P #################
# In the POD, the temporal basis comes as eigenvectors of K:
# First we compute K
print('Computing Correlation Matrix')
K = np.dot(D_Mr.transpose(), D_Mr) # Comput temporal correlation matrix
# Comput the Temporal basis Psi
Psi_P, Lambda_P, _ = np.linalg.svd(K)

####################### 3. COMPUTING THE SPATIAL STRUCTURES #################
# Observe that the POD does not need the normalization step, because 
# the energies are already the lambda^2!!!!
# Moreover, we do not need to sort the final restult since they are already ranked 
# by energy

Sigma_P=Lambda_P**0.5 # Have a look at how strong these decay !!!!

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
plt.plot(Sigma_P,'ko:')
plt.xscale('linear')
plt.yscale('linear')

# We take only the first 200 modes.
Sigma_P_t=Sigma_P[0:200]
Sigma_P_Inv_V=1/Sigma_P_t
# Accordingly we reduce psi_P
Psi_P_t=Psi_P[:,0:200]
# So we have the inverse
Sigma_P_Inv=np.diag(Sigma_P_Inv_V)


# We put some messages
print('Projecting Data')
Begin=time.time()
# This is Phi= D Psi Sigma_inv
Phi_P=np.linalg.multi_dot([D,Psi_P[:,0:200],Sigma_P_Inv])
Duration=time.time()-Begin # Measure the time of the projection
print('Decomposition completed in  '+str(Duration)+' seconds')
# Check the decomposition convergence (carefull: we have only 200 modes)
D_P=np.linalg.multi_dot([Phi_P,np.diag(Sigma_P_t),np.transpose(Psi_P_t)]) 
Error=np.linalg.norm(D-D_P)/np.linalg.norm(D)
print('Convergence Error: E_C='+"{:.2f}".format(Error/100)+' %')

"""
Energy

"""
Energy = np.zeros((len(Sigma_P),1))
Energy = Sigma_P**2/np.sum(Sigma_P**2)
cumulative = np.cumsum(Energy)/np.sum(Energy)

X_Axis = np.arange(Energy.shape[0])


fig, axes = plt.subplots(1, 2, figsize = (12,4))
ax = axes[0]
ax.bar(X_Axis, Energy, width=0.5)
ax.set_xlim(-0.25,30)
ax.set_xlabel('Modes')
ax.set_ylabel('Energy Content')

ax = axes[1]
ax.plot(cumulative, marker = 'o', markerfacecolor = 'none', markeredgecolor = 'k', ls='-', color = 'k')
ax.set_xlabel('Modes')
ax.set_ylabel('Cumulative Energy')
ax.set_xlim(0, 300)

plt.show()

# We now export the Phi, the Psi and the Psi_HAT
# So we look at the Freqs vector
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins


# Export the first r modes
Fol_Out= r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\PODmodes"
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
  
for r in range(0,10):
  print('Exporting Mode '+str(r))  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]

    # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
   # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Show The spatial structure with the spectra of the associated temporal
  # structure

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})  # 2:1 rapporto di altezza

  # First subplot
  c = ax1.contourf(Xg, Yg, Magn, cmap = 'viridis')
  cbar = plt.colorbar(c)
  
  STEPx, STEPy = 2, 2
  ax1.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k', scale=0.8)
    
  ax1.set_aspect('equal')
  ax1.set_xlabel('$x[mm]$', fontsize=14)
  ax1.set_ylabel('$y[mm]$', fontsize=14)
  ax1.set_title(f"$\\phi_{r}(x,y)$", fontsize=18)
    
  # Second subplot: FFT
  ax2.plot(Freqs, np.abs(np.fft.fft(Psi)), 'ko')
  ax2.set_xlabel('$f[Hz]$', fontsize=16)
  ax2.set_ylabel(r'$\hat{\psi}_\mathcal{F}$', fontsize=16)
  ax2.set_title(rf'$\hat{{\psi}}_{{\mathcal{{F}}{r}}}$', fontsize=18)
  ax2.set_xlim([-3, 3])
    
  # Aggiusta il layout per evitare sovrapposizioni
  plt.subplots_adjust(hspace=0.3)  # Modifica lo spazio verticale tra i subplot
    
  # Salvataggio
  #plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.pdf', dpi=100)
  plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.png', dpi=100)
    
  plt.show()
  
  
"""
Phase portrait Mode 0-1 --> Example of Characteristic pattern 
"""
mode_1_temporal = Psi_P[:,0]
mode_2_temporal = Psi_P[:,1]

plt.figure(figsize=(6, 6))  
plt.scatter(mode_1_temporal, mode_2_temporal, edgecolors='black', facecolors='none', s=20)  

plt.title('Phase portrait (0-1)', fontsize=16)
plt.xlabel('Mode 0', fontsize=14)
plt.ylabel('Mode 2', fontsize=14)

# Personalizza l'aspetto degli assi
plt.grid(True)  # Aggiungi la griglia
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostra il grafico
plt.tight_layout()  # Ottimizza il layout
plt.show()

"""
Phase portrait Modes 1-3: Example of No characteristic pattern
"""
mode_1_temporal = Psi_P[:,1]
mode_3_temporal = Psi_P[:,3]

plt.figure(figsize=(6, 6))  
plt.scatter(mode_1_temporal, mode_3_temporal, edgecolors='black', facecolors='none', s=20)  

plt.title('Phase portrait (1-3)', fontsize=16)
plt.xlabel('Mode 1', fontsize=14)
plt.ylabel('Mode 3', fontsize=14)

# Personalizza l'aspetto degli assi
plt.grid(True)  # Aggiungi la griglia
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostra il grafico
plt.tight_layout()  # Ottimizza il layout
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import time # Usefull for timing the code
import pandas as pd
import os 
"""

Same thing as before but applied to the 16kV case and Humidity

"""
# Prepare the folder with the unzipped data
FOLDER=r"D:\PIV RM\PostProcessing\7 Feb\Plasma_417dt_2vel_16kV_Humidity\GIF\EveryFrame"

# For info, these data were collected with the following parameters
n_t=500 # number of steps.
Fs=5 # Sampling frequency
dt=1/Fs # This data was sampled at 2kHz.
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'PIVlab_%04d'%5+'.txt' # Check it out: print(Name)
# Read data from a file
# DATA = pd.read_csv(Name, delimiter=',') # Here we have the four colums
Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
nxny=Dat.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s=2*nxny
## 1. Reconstruct Mesh from file
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
# Number of n_X/n_Y from forward differences
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
# Reshaping the grid from the data
n_x=(nxny//(n_y)) # Carefull with integer and float!
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y)))

####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################

for k in range(1,n_t+1):
  Name=FOLDER+os.sep+'PIVlab_%04d'%k+'.txt' # Name of the file to read
  # Read data from a file
  Dat = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
  V_X=np.array(Dat['u [m/s]']) # U component
  V_Y=np.array(Dat['v [m/s]']) # V component
  D[:,k-1]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k)+'/'+str(n_t)) 

  
# For a stationary test case like this, you might want to remove the mean  
D_MEAN=np.mean(D,1) # Temporal average (along the columns)
D_Mr=D-np.array([D_MEAN,]*n_t).transpose() # Mean Removed

# Check the mean flow
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
# Put both components as fields in the grid
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
# Or you can plot it as streamlines
contour = plt.contourf(Xg,Yg,Magn, cmap='viridis')
cbar = plt.colorbar(contour)

# One possibility is to use quiver
STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 16 kV - Humidity',fontsize=18)
    
####################### 2. CONSTRUCT THE BASIS PSI_P #################
# In the POD, the temporal basis comes as eigenvectors of K:
# First we compute K
print('Computing Correlation Matrix')
K = np.dot(D_Mr.transpose(), D_Mr) # Comput temporal correlation matrix
# Comput the Temporal basis Psi
Psi_P, Lambda_P, _ = np.linalg.svd(K)

####################### 3. COMPUTING THE SPATIAL STRUCTURES #################

Sigma_P=Lambda_P**0.5 # Have a look at how strong these decay !!!!

# Show The Mean Flow
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
plt.plot(Sigma_P,'ko:')
plt.xscale('linear')
plt.yscale('linear')

# We take only the first 200 modes.
Sigma_P_t=Sigma_P[0:200]
Sigma_P_Inv_V=1/Sigma_P_t
# Accordingly we reduce psi_P
Psi_P_t=Psi_P[:,0:200]
# So we have the inverse
Sigma_P_Inv=np.diag(Sigma_P_Inv_V)


# We put some messages
print('Projecting Data')
Begin=time.time()
# This is Phi= D Psi Sigma_inv
Phi_P=np.linalg.multi_dot([D,Psi_P[:,0:200],Sigma_P_Inv])
Duration=time.time()-Begin # Measure the time of the projection
print('Decomposition completed in  '+str(Duration)+' seconds')
# Check the decomposition convergence (carefull: we have only 200 modes)
D_P=np.linalg.multi_dot([Phi_P,np.diag(Sigma_P_t),np.transpose(Psi_P_t)]) 
Error=np.linalg.norm(D-D_P)/np.linalg.norm(D)
print('Convergence Error: E_C='+"{:.2f}".format(Error/100)+' %')

"""
Energy

"""
Energy = np.zeros((len(Sigma_P),1))
Energy = Sigma_P**2/np.sum(Sigma_P**2)
cumulative = np.cumsum(Energy)/np.sum(Energy)

X_Axis = np.arange(Energy.shape[0])


fig, axes = plt.subplots(1, 2, figsize = (12,4))
ax = axes[0]
ax.bar(X_Axis, Energy, width=0.5)
ax.set_xlim(-0.25,30)
ax.set_xlabel('Modes')
ax.set_ylabel('Energy Content')

ax = axes[1]
ax.plot(cumulative, marker = 'o', markerfacecolor = 'none', markeredgecolor = 'k', ls='-', color = 'k')
ax.set_xlabel('Modes')
ax.set_ylabel('Cumulative Energy')
ax.set_xlim(0, 300)

plt.show()

# We now export the Phi, the Psi and the Psi_HAT
# So we look at the Freqs vector
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins


# Export the first r modes
Fol_Out= r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\PODmodes"
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
  
for r in range(0,10):
  print('Exporting Mode '+str(r))  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]

    # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
   # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Show The spatial structure with the spectra of the associated temporal
  # structure

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})  # 2:1 rapporto di altezza

  # First subplot
  c = ax1.contourf(Xg, Yg, Magn, cmap = 'viridis')
  cbar = plt.colorbar(c)
  STEPx, STEPy = 2, 2
  ax1.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k', scale=0.8)
    
  ax1.set_aspect('equal')
  ax1.set_xlabel('$x[mm]$', fontsize=14)
  ax1.set_ylabel('$y[mm]$', fontsize=14)
  ax1.set_title(f"$\\phi_{r}(x,y)$", fontsize=18)
    
  # Second subplot: FFT
  ax2.plot(Freqs, np.abs(np.fft.fft(Psi)), 'ko')
  ax2.set_xlabel('$f[Hz]$', fontsize=16)
  ax2.set_ylabel(r'$\hat{\psi}_\mathcal{F}$', fontsize=16)
  ax2.set_title(rf'$\hat{{\psi}}_{{\mathcal{{F}}{r}}}$', fontsize=18)
  ax2.set_xlim([-3, 3])
    
  # Aggiusta il layout per evitare sovrapposizioni
  plt.subplots_adjust(hspace=0.3)  # Modifica lo spazio verticale tra i subplot
    
  # Salvataggio
  #plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.pdf', dpi=100)
  plt.savefig(Fol_Out + os.sep + f'POD_Mode_{r}.png', dpi=100)
    
  plt.show()
  
for i in range(0,10):
    for j in range(0,10):
        mode_1_temporal = Psi_P[:,i]
        mode_2_temporal = Psi_P[:,j]
        
        plt.figure(figsize=(6, 6))  
        plt.scatter(mode_1_temporal, mode_2_temporal, edgecolors='black', facecolors='none', s=20)  

        plt.title(f'Phase portrait ({i}-{j})', fontsize=16)
        plt.xlabel(f'Mode {i}', fontsize=14)
        plt.ylabel(f'Mode {j}', fontsize=14)

        # Personalizza l'aspetto degli assi
        plt.grid(True)  # Aggiungi la griglia
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Mostra il grafico
        plt.tight_layout()  # Ottimizza il layout
        plt.show()
        
