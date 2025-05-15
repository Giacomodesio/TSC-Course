"""
Created on 01 March 2025

@author: De Sio Giacomo

@ Title: Proper Ortoghonal Decomposition applied to PIV data from 
         my first experimental campaign

"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import ReadFile, D_matrix, POD_setup, plot_spatial_mode, plot_phase_portrait

## Parameters
n_t = 500                               # Time snapshot
Fs = 5                                  # Sampling frequency (Hz)
dt = 1/Fs 
t = np.linspace(0,dt*(n_t-1),n_t)       # Time axis 



#%%
"""
CASE N°1
"""
#------- PLASMA 16 kV - NO HUMIDITY -------------

FOLDER=r"D:\PIV RM\PostProcessing\5 Feb\Plasma_417dt_2vel_16kV\EveryFrame" 

## Reconstruct Mesh from a random snapshot 
Dat = ReadFile(FOLDER, 5)               # The fitfh snapshot is used to create the grid
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
#ax.set_xlim(0, 300)

plt.show()

####################### MODES VISUALIZATION #################
# Here the first six modes are plotted, in particular only the spatial structure of each mode is plotted

# First six modes visualization (only spatial structure)
for r in range(0,6):
    plot_spatial_mode(Phi_P, mode_index=r, Xg=Xg, Yg=Yg, n_x=n_x, n_y=n_y)


####################### PHASE PORTRAIT #################

"""
Phase portrait Mode 0-1 --> Example of Characteristic pattern 
"""
plot_phase_portrait(Psi_P, mode_x = 0, mode_y = 1)


"""
Phase portrait Modes 1-3: Example of No characteristic pattern
"""
plot_phase_portrait(Psi_P, mode_x=1, mode_y=3)



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

plt.show()

####################### MODES VISUALIZATION #################
# Here the first six modes are plotted, in particular only the spatial structure of each mode is plotted

# First six modes visualization
for r in range(0,6):
    plot_spatial_mode(Phi_P, mode_index=r, Xg=Xg, Yg=Yg, n_x=n_x, n_y=n_y)
  
####################### PHASE PORTRAIT #################

"""
Phase portrait Mode 0-1 --> Example of Characteristic pattern 
"""
plot_phase_portrait(Psi_P, mode_x = 0, mode_y = 1)

"""
Phase portrait Modes 1-3: Example of No characteristic pattern
"""
plot_phase_portrait(Psi_P, mode_x = 1, mode_y = 3)
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

plt.show()

####################### MODES VISUALIZATION #################
# Here the first six modes are plotted, in particular only the spatial structure of each mode is plotted

# First six modes visualization
for r in range(0,6):
    plot_spatial_mode(Phi_P, mode_index=r, Xg=Xg, Yg=Yg, n_x=n_x, n_y=n_y)

####################### PHASE PORTRAIT #################

for i in range(0,6):
    for j in range(0,6):
        plot_phase_portrait(Psi_P, mode_x = i, mode_y = j)
        


