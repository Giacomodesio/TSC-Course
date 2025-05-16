import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd
import os


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
    D_Mr : Data matrix without the mean of dimensions (2*n_s, n_t), where n_s 
           is the number of spatial points and n_t is the number of time steps.
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

    Sigma_P=Lambda_P**0.5 # Amplitude of the mode

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

    Phi_P=np.linalg.multi_dot([D_Mr,Psi_P[:,0:200],Sigma_P_Inv])

    Duration=time.time() - Begin 
    print('Decomposition completed in  '+str(Duration)+' seconds')

    # Decomposition convergence (only 200 modes!)
    D_P=np.linalg.multi_dot([Phi_P,np.diag(Sigma_P_t),np.transpose(Psi_P_t)]) 
    Error=np.linalg.norm(D_Mr-D_P)/np.linalg.norm(D_Mr)
    print('Convergence Error: E_C='+"{:.2f}".format(Error/100)+' %')
    
    return Psi_P, Lambda_P, Sigma_P, Phi_P


def plot_spatial_mode(Phi_P, mode_index, Xg, Yg, n_x, n_y, step_x=2, step_y=2):
    """
    Plots the spatial structure of a specific POD mode.

    Parameters:
    -----------
    Phi_P : ndarray (2*nx*ny x n_modes)
        Spatial modes matrix (concatenated Vx and Vy components).
    mode_index : int
        Index of the mode to plot.
    Xg, Yg : 2D arrays
        Meshgrid arrays for the spatial domain.
    n_x, n_y : int
        Number of grid points in x and y directions.
    step_x, step_y : int
        Steps for downsampling the quiver vectors.
    """
    nxny = n_x * n_y

    print(f'Exporting Mode {mode_index}') 

    Phi = Phi_P[:, mode_index]
    V_X_m = Phi[0:nxny]
    V_Y_m = Phi[nxny:]

    # Compute magnitude
    Mod = np.sqrt(V_X_m**2 + V_Y_m**2)

    # Reshape for plotting
    Vxg = V_X_m.reshape((n_x, n_y)).T
    Vyg = V_Y_m.reshape((n_x, n_y)).T
    Magn = Mod.reshape((n_x, n_y)).T

    # Plot spatial structure
    plt.figure(figsize=(8, 6))

    # Contour plot
    c = plt.contourf(Xg, Yg, Magn, cmap='viridis')
    plt.colorbar(c)

    # Quiver plot (with inversion of y-component if needed)
    plt.quiver(Xg[::step_x, ::step_y], Yg[::step_x, ::step_y], 
               Vxg[::step_x, ::step_y], -Vyg[::step_x, ::step_y], 
               color='k', scale=0.8)

    plt.gca().set_aspect('equal')
    plt.xlabel('$x[mm]$', fontsize=14)
    plt.ylabel('$y[mm]$', fontsize=14)
    plt.title(f"$\\phi_{mode_index}(x,y)$", fontsize=18)

    plt.tight_layout()
    plt.show()




def plot_phase_portrait(Psi_P, mode_x, mode_y, title=None):
    """
    Plots a phase portrait between two temporal modes of the matrix Psi_P.

    Parameters:
    -----------
    Psi_P : ndarray (n_snapshots x n_modes)
        Temporal coefficients matrix.
    mode_x : int
        Index of the mode for the x-axis (zero-based index).
    mode_y : int
        Index of the mode for the y-axis (zero-based index).
    title : str, optional
        Title of the plot. If None, a default title is used.
    """

    mode_x_temporal = Psi_P[:, mode_x]
    mode_y_temporal = Psi_P[:, mode_y]

    plt.figure(figsize=(6, 6))  
    plt.scatter(mode_x_temporal, mode_y_temporal, 
                edgecolors='black', facecolors='none', s=20)  

    if title is None:
        title = f'Phase portrait ({mode_x}-{mode_y})'

    plt.title(title, fontsize=16)
    plt.xlabel(f'Mode {mode_x}', fontsize=14)
    plt.ylabel(f'Mode {mode_y}', fontsize=14)
    plt.grid(True)  
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

