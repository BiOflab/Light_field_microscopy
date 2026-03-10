import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import tifffile


# ==========================================
# --- CONFIGURATION PARAMETERS (EDIT HERE) ---
# ==========================================

# OPTICAL SETUP
M_nominal = 20.0        # Grossissement écrit sur l'objectif
NA = 0.75               # Ouverture Numérique
f_tube_ref = 200.0      # Focale tube de référence (mm)
f_tube_actual = 330.0   # VOTRE lentille de tube réelle (mm)

# MEDIUM & LIGHT
n_medium = 1.0          # Milieu (1.0 pour Air)
wl_nm = 530.0           # Longueur d'onde (nm)

# POSITION DE L'OBJET (Défocus & Latéral)
z_obj_um = 0.0          # Défocus axial (microns)

x_obj_um = 0.0          # Décalage latéral en X (microns) 
y_obj_um = 0.0          # Décalage latéral en Y (microns) 

# SCANNING Z (STACK SETTINGS) 
z_start_um = -25.0      # Z de départ
z_end_um = 25.0         # Z de fin
z_step_um = 2.5         # Pas de scan (step)

save_filename = f"PSF_zStack_(0,0)_pix7.1.tiff"

# MICROLENS ARRAY (MLA)
pitch_ml_um = 100.0     # Pitch de la MLA (microns)
f_ml_mm = 2.1           # Focale Microlentille (mm)

# SIMULATION SETTINGS
sim_pixel_size_um = 7.1 # Résolution de calcul (microns/pixel)
N_ml_sim = 30          # Nombre de microlentilles simulées (NxN)



# ==========================================
# --- PHYSICS MODELS ---
# ==========================================

# --- 1. MODÈLE DE CHAMP INCIDENT (DEBYE) ---
def calculate_incident_field_debye(
    grid_size_pixels,
    pixel_size_effective_um,
    wavelength_nm,
    NA,
    n_medium,
    magnification,
    z_object_offset_um,
    x_object_offset_um=0,
    y_object_offset_um=0,  # <--- Ajout du paramètre y_object_offset_um
    aberrations_zernike=None
):
    """
    Calcule le CHAMP COMPLEXE (Amplitude + Phase) incident sur la MLA.
    """
    # Constantes
    lambda_m = wavelength_nm * 1e-9
    k = 2 * np.pi / lambda_m
    
    # Grille spatiale au niveau de la MLA
    x = (np.arange(grid_size_pixels) - grid_size_pixels//2) * pixel_size_effective_um * 1e-6
    X, Y = np.meshgrid(x, x)

    # Coordinates relative to the Geometric Image Point
    # Si l'objet se décale de (x_obj, y_obj), l'image se décale de (-M * x_obj, -M * y_obj)
    x_img_shift_m = -x_object_offset_um * 1e-6 * magnification
    y_img_shift_m = -y_object_offset_um * 1e-6 * magnification # <--- Calcul du décalage image en Y
    
    # Shifted Radial Coordinates for the Bessel term
    # Mise à jour pour inclure le décalage Y
    R_shifted = np.sqrt((X - x_img_shift_m)**2 + (Y - y_img_shift_m)**2) 
    
    # --- OPTIQUE GÉOMÉTRIQUE (OBJECTIF + TUBE LENS) ---
    NA_image = NA / magnification
    alpha_max = np.arcsin(NA_image / n_medium) 
    z_image_defocus_m = (z_object_offset_um * 1e-6) * (magnification**2)
    
    # --- INTÉGRALE DE DEBYE ---
    N_alpha = 50
    alpha = np.linspace(0, alpha_max, N_alpha)
    d_alpha = alpha[1] - alpha[0]
    
    U_inc = np.zeros_like(R_shifted, dtype=complex)
    
    # Pré-calcul des aberrations (laissé pour référence)
    def get_aberration_phase(rho_norm):
        phi = np.zeros_like(rho_norm)
        if aberrations_zernike:
             if 9 in aberrations_zernike:
                 phi += aberrations_zernike[9] * np.sqrt(5) * (6*rho_norm**4 - 6*rho_norm**2 + 1)
        return phi

    # print(f"Calcul Champ Debye (NA_img={NA_image:.4f}, z_img_defocus={z_image_defocus_m*1e3:.3f}mm)...")
    
    for i in range(N_alpha):
        a = alpha[i]
        
        # A. Defocus Phase
        defocus_phase = np.exp(1j * k * z_image_defocus_m * np.cos(a))
        
        # B. Apodization
        apodization = np.sqrt(np.cos(a))
        
        # C. Radial Term (J0)
        bessel_term = jv(0, k * R_shifted * np.sin(a))
        
        # Phase d'aberration
        rho_norm = np.sin(a) / np.sin(alpha_max)
        aberration_phase = np.exp(1j * get_aberration_phase(rho_norm))
        
        integrand = apodization * defocus_phase * aberration_phase * bessel_term * np.sin(a) * d_alpha
        U_inc += integrand
        
    return U_inc

# --- 2. MODÈLE DE MLA (MASQUE DE PHASE) ---
def apply_mla_transmission(U_field, grid_size_pixels, pixel_size_um, pitch_um, f_ml_mm, wavelength_nm):
    lambda_m = wavelength_nm * 1e-9
    k = 2 * np.pi / lambda_m
    f_ml_m = f_ml_mm * 1e-3
    pitch_m = pitch_um * 1e-6
    
    x = (np.arange(grid_size_pixels) - grid_size_pixels//2) * pixel_size_um * 1e-6
    X, Y = np.meshgrid(x, x)
    
    # Coordonnées locales périodiques (Tiling)
    X_local = (X + pitch_m/2) % pitch_m - pitch_m/2
    Y_local = (Y + pitch_m/2) % pitch_m - pitch_m/2
    R_local_sq = X_local**2 + Y_local**2
    
    # Lentille mince convergente
    mla_phase = np.exp(-1j * k / (2 * f_ml_m) * R_local_sq)
    
    return U_field * mla_phase

# --- 3. PROPAGATION DE FRESNEL ---
def propagate_to_sensor(U_mla, grid_size_pixels, pixel_size_um, distance_mm, wavelength_nm):
    lambda_m = wavelength_nm * 1e-9
    z_m = distance_mm * 1e-3
    
    fx = np.fft.fftfreq(grid_size_pixels, d=pixel_size_um*1e-6)
    FX, FY = np.meshgrid(fx, fx)
    
    # Transfer Function
    H = np.exp(-1j * np.pi * lambda_m * z_m * (FX**2 + FY**2))
    
    U_fft = fft2(ifftshift(U_mla)) 
    U_prop_fft = U_fft * H
    U_sensor = fftshift(ifft2(U_prop_fft)) 
    
    return U_sensor

# ==========================================
# --- MAIN SIMULATION LOGIC ---
# ==========================================

def simulate_light_field_psf_wave_optics():
    
    # --- 1. CALCULATE DERIVED PARAMETERS ---
    M_effective = M_nominal * (f_tube_actual / f_tube_ref)
    
    grid_size_pixels = int((N_ml_sim * pitch_ml_um) / sim_pixel_size_um)
    if grid_size_pixels % 2 != 0: grid_size_pixels += 1
    
    print(f"\n--- Optical Configuration ---")
    print(f"-> Effective Mag: {M_effective:.2f}x")
    print(f"-> Source Position (x, y, z): ({x_obj_um}, {y_obj_um}, {z_obj_um}) µm")

    # --- A. Champ Incident (Debye) ---
    U_inc = calculate_incident_field_debye(
        grid_size_pixels, sim_pixel_size_um, wl_nm, 
        NA, n_medium, M_effective, z_obj_um, 
        x_object_offset_um=x_obj_um, 
        y_object_offset_um=y_obj_um # <--- Passation du nouveau paramètre
    )
    
    # --- B. Passage through MLA ---
    U_after_mla = apply_mla_transmission(
        U_inc, grid_size_pixels, sim_pixel_size_um, 
        pitch_ml_um, f_ml_mm, wl_nm
    )
    
    # --- C. Propagation to Sensor ---
    U_sensor = propagate_to_sensor(
        U_after_mla, grid_size_pixels, sim_pixel_size_um, 
        f_ml_mm, wl_nm
    )
    
    # Intensity
    I_sensor = np.abs(U_sensor)**2
    I_sensor /= np.max(I_sensor) 
    
    # --- D. AFFICHAGE ---
    
    total_width_um = grid_size_pixels * sim_pixel_size_um
    half_width = total_width_um / 2.0
    extent = [-half_width, half_width, -half_width, half_width]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Incident Field
    ax1 = axes[0]
    im1 = ax1.imshow(np.abs(U_inc)**2, extent=extent, cmap='inferno', origin='lower')
    ax1.set_title(f'Incident Field at Image Plane\nZ_img={z_obj_um * M_effective**2 / 1000:.1f}mm')
    ax1.set_xlabel('Position (µm)')
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Sensor PSF
    ax2 = axes[1]
    I_log = np.clip(I_sensor, 1e-4, 1.0)
    
    im2 = ax2.imshow(I_log, extent=extent, cmap='inferno', origin='lower',
                     norm=plt.Normalize(vmin=1e-4, vmax=1.0))
    ax2.set_title('PSF on the Sensor (after MLA) [Log Scale]')
    ax2.set_xlabel('Position (µm)')
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    
    # Visual Grid for Lenslets
    ticks_ml = np.arange(-N_ml_sim//2, N_ml_sim//2 + 1) * pitch_ml_um
    ticks_ml = ticks_ml[(ticks_ml >= -half_width) & (ticks_ml <= half_width)]
    
    ax2.set_xticks(ticks_ml)
    ax2.set_yticks(ticks_ml)
    ax2.grid(color='cyan', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# --- MAIN Z-SCAN LOGIC ---
# ==========================================

def simulate_z_stack():
    
    # --- 1. PREPARATION ---
    M_effective = M_nominal * (f_tube_actual / f_tube_ref)
    grid_size_pixels = int((N_ml_sim * pitch_ml_um) / sim_pixel_size_um)
    if grid_size_pixels % 2 != 0: grid_size_pixels += 1
    
    # Création du vecteur Z
    z_values = np.arange(z_start_um, z_end_um + z_step_um, z_step_um)
    num_slices = len(z_values)
    
    print(f"\n--- Starting Z-Stack Simulation ---")
    print(f"X, Y Position: ({x_obj_um}, {y_obj_um}) µm")
    print(f"Z Range: {z_start_um} to {z_end_um} µm (Step: {z_step_um} µm)")
    print(f"Total Slices: {num_slices}")
    print(f"Grid Size: {grid_size_pixels}x{grid_size_pixels}")
    
    # Initialisation du Stack 3D (Z, Y, X) en float32 pour économiser la mémoire
    image_stack = np.zeros((num_slices, grid_size_pixels, grid_size_pixels), dtype=np.float32)
    
    # --- 2. BOUCLE DE SCAN ---
    for i, current_z in enumerate(z_values):
        print(f"Processing Slice {i+1}/{num_slices} : Z = {current_z:.1f} µm")
        
        # A. Champ Incident (Change à chaque Z)
        U_inc = calculate_incident_field_debye(
            grid_size_pixels, sim_pixel_size_um, wl_nm, 
            NA, n_medium, M_effective, current_z, 
            x_object_offset_um=x_obj_um, 
            y_object_offset_um=y_obj_um
        )
        
        # B. MLA (Identique, mais appliqué au nouveau champ)
        U_after_mla = apply_mla_transmission(
            U_inc, grid_size_pixels, sim_pixel_size_um, 
            pitch_ml_um, f_ml_mm, wl_nm
        )
        
        # C. Propagation (Identique)
        U_sensor = propagate_to_sensor(
            U_after_mla, grid_size_pixels, sim_pixel_size_um, 
            f_ml_mm, wl_nm
        )
        
        # D. Intensité
        I_slice = np.abs(U_sensor)**2
        
        # Stockage dans le stack
        image_stack[i, :, :] = I_slice

    # --- 3. NORMALISATION & SAUVEGARDE ---
    print("\nSimulation Finished. Saving...")
    
    # Normalisation Globale (Optionnelle mais recommandée pour comparer les plans Z)
    # On normalise par le pixel le plus brillant de TOUT le stack
    max_val = np.max(image_stack)
    if max_val > 0:
        image_stack /= max_val
        print(f"Stack normalized by global max: {max_val:.4e}")
    
    # Sauvegarde TIFF (float32 pour précision scientifique)
    tifffile.imwrite(save_filename, image_stack)
    print(f"Saved stack to: {save_filename}")

    
    


if __name__ == "__main__":
    # simulate_light_field_psf_wave_optics()
    simulate_z_stack()
