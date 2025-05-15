"""
Created on 01 March 2025

@author: De Sio Giacomo

@ Title: POD applied to the plasma 16 kV no Humidity

"""

import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd
import os 

## Parameters
n_t = 500                               # Time snapshot
Fs = 5                                  # Sampling frequency (Hz)
dt = 1/Fs 
t = np.linspace(0,dt*(n_t-1),n_t)       # Time axis 

def ReadFile(folder, snapshot_number):
    
    """
    Parameters
    ----------
    folder : Name of the folder that contains all snapshot files.
    snapshot_number : Time step you want to analyse

    Returns
    -------
    Data: Information contained in each snapshot

    """
    
    Name=folder+os.sep+'PIVlab_%04d'%snapshot_number+'.txt'
    Data = pd.read_csv(Name, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna()
    
    return Data

def D_matrix(FOLDER, n_s, n_t):
    
    """
    Parameters
    ----------
    FOLDER : Name of the folder that contains all snapshot files.
    n_s : Number of spatial points (grid points per snapshot).
    n_t : Number of time steps (snapshots to process).

    Returns
    -------
    D : Data matrix of dimensions (2*n_s, n_t) containing the velocity fields.
        For each time step, the U and V velocity components are stacked column-wise
        to build the snapshot matrix used in data-driven analysis methods.
    """
    
    ####################### CONSTRUCT THE DATA MATRIX D #################
    # Initialize the data matrix D
    D=np.zeros([n_s,n_t])                   # See Lecture 3 DDMA
    #####################################################################

    for k in range(1,n_t+1):    
      Dat = ReadFile(FOLDER, k)
      V_X=np.array(Dat['u [m/s]'])          # U component
      V_Y=np.array(Dat['v [m/s]'])          # V component
      D[:,k-1]=np.concatenate([V_X,V_Y],axis=0) 
      print('Loading Step '+str(k)+'/'+str(n_t)) 
      
    return D

def POD_setup(D_Mr):
    
    """
    Parameters
    ----------
    D_Mr : Reduced data matrix (usually spatially filtered or preprocessed velocity fields)
           of dimensions (2*n_s, n_t), where n_s is the number of spatial points
           and n_t is the number of time steps.

    Returns
    -------
    Psi_P : Temporal modes (right singular vectors from SVD).
    Lambda_P : Squared singular values (eigenvalues of the temporal correlation matrix).
    Sigma_P : Singular values (square root of Lambda_P), representing mode amplitudes.
    Phi_P : Spatial modes (left singular vectors), obtained by projecting data onto temporal modes.
    
    Description
    -----------
    This function performs a Proper Orthogonal Decomposition (POD) on the input data matrix D_Mr.
    The process involves:
    
    1. Constructing the temporal correlation matrix K and computing its SVD to obtain temporal modes.
    2. Computing the amplitude of each mode (Sigma_P) from the singular values.
    3. Projecting the data onto the temporal modes to extract the spatial modes (Phi_P).
    
    The function also plots the energy decay of the modes and estimates the reconstruction error
    after truncating the decomposition to 200 modes.
    """
    
    
    ####################### 1. CONSTRUCT THE TEMPORAL STRUCTURE PSI_P #################
    # In the POD, the temporal basis comes as eigenvectors of K.
    # First we compute K: temporal correlation matrix

    print('Computing Correlation Matrix')

    K = np.dot(D_Mr.transpose(), D_Mr)        # DDMA Lecture 5
    Psi_P, Lambda_P, _ = np.linalg.svd(K)     # Single Value Decomposition to compute temporal and spatial structures


    ####################### 2. COMPUTING THE AMPLITUDE SIGMA_P #################

    Sigma_P=Lambda_P**0.5 # Spatial structure

    # Decay of the energy associated to every mode
    fig, ax = plt.subplots(figsize=(8, 5)) 
    plt.plot(Sigma_P,'ko:')
    plt.xlabel('Mode Number', fontsize = 18)
    plt.ylabel('Mode Energy', fontsize = 18)
    plt.xscale('linear')
    plt.yscale('linear')

    ####################### 3. COMPUTING THE SPATIAL STRUCTURE PHI_P #################

    # According to DDMA lecture 5: D = PHI_P * SIGMA_P * PSI_P.T .
    # So, PHI_P = D * PSI_P * inv(SIGMA_P)

    # Limit to 200 modes
    Sigma_P_t = Sigma_P[0:200]
    Sigma_P_Inv_V = 1/Sigma_P_t
    Psi_P_t = Psi_P[:,0:200]
    Sigma_P_Inv=np.diag(Sigma_P_Inv_V)      # Diagonal, squared matrix to simplify calculation

    # Now I have D, PSI_P and inv(SIGMA_P)
    print('Projecting Data')
    Begin=time.time()

    Phi_P=np.linalg.multi_dot([D,Psi_P[:,0:200],Sigma_P_Inv])

    Duration=time.time() - Begin 
    print('Decomposition completed in  '+str(Duration)+' seconds')

    # Decomposition convergence (only 200 modes!)
    D_P=np.linalg.multi_dot([Phi_P,np.diag(Sigma_P_t),np.transpose(Psi_P_t)]) 
    Error=np.linalg.norm(D-D_P)/np.linalg.norm(D)
    print('Convergence Error: E_C='+"{:.2f}".format(Error/100)+' %')
    
    return Psi_P, Lambda_P, Sigma_P, Phi_P

#%%
"""
CASE N°1
"""
#------- PLASMA 16 kV - NO HUMIDITY -------------

FOLDER=r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\EveryFrame" 

## Reconstruct Mesh from a random snapshot 
Dat = ReadFile(FOLDER, 5)
nxny=Dat.shape[0] 
n_s=2*nxny                              # Doubled bc of the two velocity components
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);              
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; 
n_y=DAT[0]+1;
n_x=(nxny//(n_y))                       # Number of n_x/n_y from forward differences
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y))) # Final mesh --> 121 x 134.

## D matrix
D = D_matrix(FOLDER, n_s, n_t)  

## STATIONARY FLOW --> Mean removal 
D_MEAN = np.mean(D,1)                     # Temporal average (along rows)
D_Mr = D - np.array([D_MEAN,]*n_t).transpose() 

####################### MEAN FLOW VISUALIZATION ##################

# Magnitude
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y))) # Vector --> 2D matrix
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))


fig, ax = plt.subplots(figsize=(10, 6)) 
contour = plt.contourf(Xg, Yg, Magn, cmap='viridis')
cbar = plt.colorbar(contour)

STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') 
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 16 kV - No humidity',fontsize=18)


####################### POD DECOMPOSITION #########################

Psi_P, Lambda_P, Sigma_P, Phi_P = POD_setup(D_Mr)

####################### CUMULATIVE ENERGY CONTENT #################

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

####################### MODES VISUALIZATION #################

# To compute the frequency content
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins

# First six modes visualization
for r in range(0,6):
    
  print('Exporting Mode '+str(r)) 
  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]
  Sigma = Sigma_P[r]

  # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
  # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Spatial structure with the spectra of the associated temporal structure
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

  plt.subplots_adjust(hspace=0.3)  # Vertical space between the two plots
  plt.show()

####################### PHASE PORTRAIT #################

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
plt.grid(True)  
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
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
plt.grid(True)  
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()




#%%

"""
CASE N°2
"""
#------- PLASMA 17 kV - NO HUMIDITY -------------

FOLDER=r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_17kV\EveryFrame"

## Reconstruct Mesh from a random snapshot 
Dat = ReadFile(FOLDER, 5)
nxny=Dat.shape[0] 
n_s=2*nxny                              # Doubled bc of the two velocity components
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);              
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; 
n_y=DAT[0]+1;
n_x=(nxny//(n_y))                       # Number of n_x/n_y from forward differences
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y))) # Final mesh --> 121 x 134.

## D matrix
D = D_matrix(FOLDER, n_s, n_t)  

## STATIONARY FLOW --> Mean removal 
D_MEAN = np.mean(D,1)                     # Temporal average (along rows)
D_Mr = D - np.array([D_MEAN,]*n_t).transpose() 

####################### MEAN FLOW VISUALIZATION #########

# Magnitude
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y))) # Vector --> 2D matrix
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))


fig, ax = plt.subplots(figsize=(10, 6)) 
contour = plt.contourf(Xg, Yg, Magn, cmap='viridis')
cbar = plt.colorbar(contour)

STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') 
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 17 kV - No humidity',fontsize=18)


####################### POD DECOMPOSITION ######

Psi_P, Lambda_P, Sigma_P, Phi_P = POD_setup(D_Mr)

####################### CUMULATIVE ENERGY CONTENT #################

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

####################### MODES VISUALIZATION #################

# To compute the frequency content
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins

# First six modes visualization
for r in range(0,6):
    
  print('Exporting Mode '+str(r)) 
  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]
  Sigma = Sigma_P[r]

  # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
  # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Spatial structure with the spectra of the associated temporal structure
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

  plt.subplots_adjust(hspace=0.3)  # Vertical space between the two plots
  plt.show()
  
####################### PHASE PORTRAIT #################

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
plt.grid(True)  
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
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
plt.grid(True)  
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#%%
"""
CASE N°3
"""
#------- PLASMA 16 kV - WITH HUMIDITY -------------

FOLDER=r"D:\PIV RM\PostProcessing\7 Feb\Plasma_417dt_2vel_16kV_Humidity\GIF\EveryFrame"

## Reconstruct Mesh from a random snapshot 
Dat = ReadFile(FOLDER, 5)
nxny=Dat.shape[0] 
n_s=2*nxny                              # Doubled bc of the two velocity components
X_S=np.array(Dat['x [m]'])
Y_S=np.array(Dat['y [m]'])
GRAD_X=np.diff(X_S); 
GRAD_Y=np.diff(Y_S);              
IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; 
n_y=DAT[0]+1;
n_x=(nxny//(n_y))                       # Number of n_x/n_y from forward differences
Xg=np.transpose(X_S.reshape((n_x,n_y)))
Yg=np.transpose(Y_S.reshape((n_x,n_y))) # Final mesh --> 121 x 134.

## D matrix
D = D_matrix(FOLDER, n_s, n_t)  

## STATIONARY FLOW --> Mean removal 
D_MEAN = np.mean(D,1)                     # Temporal average (along rows)
D_Mr = D - np.array([D_MEAN,]*n_t).transpose() 

####################### MEAN FLOW VISUALIZATION #########

# Magnitude
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
Mod=np.sqrt(V_X_m**2+V_Y_m**2)
Vxg=np.transpose(V_X_m.reshape((n_x,n_y))) # Vector --> 2D matrix
Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
Magn=np.transpose(Mod.reshape((n_x,n_y)))


fig, ax = plt.subplots(figsize=(10, 6)) 
contour = plt.contourf(Xg, Yg, Magn, cmap='viridis')
cbar = plt.colorbar(contour)

STEPx=2
STEPy=2
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot

ax.set_aspect('equal') 
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Flow - Plasma 16 kV - with humidity',fontsize=18)
    
####################### POD DECOMPOSITION ######

Psi_P, Lambda_P, Sigma_P, Phi_P = POD_setup(D_Mr)

####################### CUMULATIVE ENERGY CONTENT #################

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

####################### MODES VISUALIZATION #################

# To compute the frequency content
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins

# First six modes visualization
for r in range(0,6):
    
  print('Exporting Mode '+str(r)) 
  
  Phi=Phi_P[:,r]
  Psi=Psi_P[:,r]
  Sigma = Sigma_P[r]

  # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
  # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Spatial structure with the spectra of the associated temporal structure
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

  plt.subplots_adjust(hspace=0.3)  # Vertical space between the two plots
  plt.show()

####################### PHASE PORTRAIT #################

for i in range(0,6):
    for j in range(0,6):
        mode_1_temporal = Psi_P[:,i]
        mode_2_temporal = Psi_P[:,j]
        
        plt.figure(figsize=(6, 6))  
        plt.scatter(mode_1_temporal, mode_2_temporal, edgecolors='black', facecolors='none', s=20)  
        plt.title(f'Phase portrait ({i}-{j})', fontsize=16)
        plt.xlabel(f'Mode {i}', fontsize=14)
        plt.ylabel(f'Mode {j}', fontsize=14)
        plt.grid(True)  
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
        
