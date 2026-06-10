
# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('test.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

import os
import re
import zipfile
import onecode
import numpy as np
import rasterio
from onecode import Logger, Mode, Project, dropdown, text_input, file_output, file_input
from rasterio.enums import Resampling
from sklearn.decomposition import PCA


BAND_COMBINATIONS = {
    "Natural Color (4-3-2)":            (4, 3, 2),
    "False Color (5-4-3)":              (5, 4, 3),
    "Geological structure (7-5-3)":     (7, 5, 3),
    "Geological structure bis (3-4-7)": (3, 4, 7),
}

BAND_RATIOS = {
    "Ferric oxydes 4/2":              (4, 2),
    "Clays, hydroxyles minerals 6/5": (6, 5),
    "Clays, hydroxyles minerals 7/5": (7, 5),
    "Iron oxydes 4/3":                (4, 3),
    "Silice 3/1":                     (3, 1),
    "Carbonates 6/3":                 (6, 3),
    "Green vegetation 5/4":           (5, 4),
    "Clay minerals 6/7":              (6, 7),
}

BAND_COMPLEXES = {
    "Ferric Iron 4/2x(4+6)/5":             (2, 4, 6, 5, "np.where((bands[5] != 0) & (bands[2] != 0), (bands[4]/bands[2])*(bands[4]+bands[6])/bands[5], np.nan)"),
    "Ferrous Iron (3+6)/(4+5)":            (3, 6, 4, 5, "np.where((bands[4]+bands[5]) != 0, (bands[3]+bands[6])/(bands[4]+bands[5]), np.nan)"),
    "Iron Sulfate 2/1-5/4":                (2, 1, 5, 4, "np.where((bands[4] != 0) & (bands[1] != 0), (bands[2]/bands[1])-(bands[5]/bands[4]), np.nan)"),
    "Clay Sulfate Mica Marble 6/7-5/4":    (6, 7, 5, 4, "np.where((bands[7] != 0) & (bands[4] != 0), (bands[6]/bands[7])-(bands[5]/bands[4]), np.nan)"),
    "Hydrated Minerals (5-6)/(6+5)":       (5, 6, 6, 5, "np.where((bands[6]+bands[5]) != 0, (bands[5]-bands[6])/(bands[6]+bands[5]), np.nan)"),
    "Clay Alteration Minerals (6-7)/(6+7)":(6, 7, 6, 7, "np.where((bands[6]+bands[7]) != 0, (bands[6]-bands[7])/(bands[6]+bands[7]), np.nan)"),
    "Litho Discrimination (6-2)/(6+2)":    (6, 2, 6, 2, "np.where((bands[6]+bands[2]) != 0, (bands[6]-bands[2])/(bands[6]+bands[2]), np.nan)"),
    "Alteration Minerals (6-5)/(6+5)":     (6, 5, 6, 5, "np.where((bands[6]+bands[5]) != 0, (bands[6]-bands[5])/(bands[6]+bands[5]), np.nan)"),
}

BAND_ACP = { 
    "CP1": 1,
    "CP2": 2,
    "CP3": 3,
    "CP4": 4,
    "CP5": 5,
    "CP6": 6,
}




def run():
    onecode.Logger.info(f"Hello {text_input('your name', 'OneCoder')}!")

    zip_path = file_input(
        key="InputFolderZip",
        value="/path/to/landsat.zip",
        label="Dossier bandes Landsat 8",
        optional=False,
    )

    prefix = text_input(
        key="Prefix",
        value="prefix",
        label="Prefixe des fichiers",
        optional=False,
    )

    suffix = text_input(
        key="Suffix",
        value="suffix",
        label="Suffixe des fichiers",
        optional=False,
    )

    chosen_combinations = dropdown(
        key="Combinaisons",
        value=["Natural Color (4-3-2)"],
        label="Choisissez les combinaisons de bandes",
        options=[
            "Natural Color (4-3-2)",
            "False Color (5-4-3)",
            "Geological structure (7-5-3)",
            "Geological structure bis (3-4-7)",
        ],
        multiple=True,
    )

    chosen_ratios = dropdown(
        key="Ratios",
        value=["Ferric oxydes 4/2"],
        label="Choisissez les indices de ratios",
        options=[
            "Ferric oxydes 4/2",
            "Clays, hydroxyles minerals 6/5",
            "Clays, hydroxyles minerals 7/5",
            "Iron oxydes 4/3",
            "Silice 3/1",
            "Carbonates 6/3",
            "Green vegetation 5/4",
            "Clay minerals 6/7",
        ],
        multiple=True,
    )

    chosen_complexes = dropdown(
        key="Calculs algebriques",
        value=["Ferric Iron 4/2x(4+6)/5"],
        label="Choisissez les calculs algebriques de bandes",
        options=[
            "Ferric Iron 4/2x(4+6)/5",
            "Ferrous Iron (3+6)/(4+5)",
            "Iron Sulfate 2/1-5/4",
            "Clay Sulfate Mica Marble 6/7-5/4",
            "Hydrated Minerals (5-6)/(6+5)",
            "Clay Alteration Minerals (6-7)/(6+7)",
            "Litho Discrimination (6-2)/(6+2)",
            "Alteration Minerals (6-5)/(6+5)",
        ],
        multiple=True,
    )

    acp_choice = dropdown(
        key="PCA",
        value = ["CP1", "CP2", "CP3"],
        label="Choisissez les composantes principales à visualiser (3 max)",
        options = ["CP1", "CP2", "CP3","CP4", "CP5", "CP6"],
        multiple=True,
    )       

    acp_combination = text_input(
        key = "RGB PCA combination",
        value = "R=CP3, G=CP1, B=CP2",
        label = "Choisissez la combinaison RGB pour représenter les composantes principales",
        optional = False,
    )

    if zip_path and os.path.exists(zip_path):
        base_path = os.path.join(os.path.dirname(zip_path), "unzipped_bands")
        os.makedirs(base_path, exist_ok=True)

        Logger.info(f"Extraction de l'archive {os.path.basename(zip_path)}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
            Logger.info("Extraction terminée avec succès.")
        except Exception as e:
            Logger.error(f"❌ Erreur lors du dézippage: {str(e)}.")
            return
    else: 
        Logger.error("❌ Fichier zip introuvable, vérifiez le chemin d'accès.")
        return 

    traitement_image(
            base_path, prefix, suffix,
            chosen_combinations or [],
            chosen_ratios or [],
            chosen_complexes or [],
            acp_choice or [],
            acp_combination or [],
        )


def _load_bands(base_path, prefix, suffix, band_range=range(1, 8)):
    #"""Charge les bandes Landsat depuis le dossier."""
    #bands = {}
    #profile = None
    #nodata_value = -9999.0
#
    #for i in band_range:
    #    file_path = os.path.join(base_path, f"{prefix}_B{i}_{suffix}.tif")
    #    file_name = f"{prefix}_B{i}_{suffix}.tif"
#
    #    if not os.path.exists(file_path):
    #        for root, dirs, files in os.walk(base_path):
    #            if file_name in files:
    #                file_path = os.path.join(root, file_name)
    #                break
#
    #    if os.path.exists(file_path):
    #        with rasterio.open(file_path) as src:
    #            band_data = src.read(1, resampling=Resampling.nearest).astype(np.float32)
    #            band_data[band_data == nodata_value] = np.nan
    #            bands[i] = band_data
    #            profile = src.profile
    #    else:
    #        Logger.warning(f"⚠️ Fichier manquant : {file_path}")
    
    """Charge automatiquement les bandes Landsat en détectant le motif _Bi_."""
    bands = {}
    profile = None
    nodata_value = -9999.0

    # 1. On liste absolument tous les fichiers extraits (et dans les sous-dossiers)
    all_files = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                all_files.append(os.path.join(root, f))

    Logger.info(f"🔍 {len(all_files)} fichiers rasters détectés dans l'archive.")

    # 2. Pour chaque numéro de bande requis, on cherche le fichier qui contient le tag "_Bi_" ou "_Bi."
    for i in band_range:
        file_path = None
        target_pattern = f"_B{i}_"
        target_pattern_end = f"_B{i}." # Au cas où la bande finit le nom (ex: image_B1.tif)

        for path in all_files:
            filename = os.path.basename(path)
            if target_pattern in filename or target_pattern_end in filename:
                file_path = path
                break

        if file_path and os.path.exists(file_path):
            Logger.info(f"📖 Chargement réussi pour la Bande {i} : {os.path.basename(file_path)}")
            with rasterio.open(file_path) as src:
                band_data = src.read(1, resampling=Resampling.nearest).astype(np.float32)
                band_data[band_data == nodata_value] = np.nan
                bands[i] = band_data
                profile = src.profile
        else:
            Logger.warning(f"⚠️ Impossible de trouver un fichier pour la Bande {i} dans l'archive.")

    return bands, profile


def _normalize(image):
    """Normalisation percentile 2-98."""
    normalized = np.zeros_like(image)
    for idx in range(image.shape[0]):
        band = image[idx]
        vmin, vmax = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
        normalized[idx] = np.clip((band - vmin) / (vmax - vmin) * 65535, 0, 65535)
    return normalized.astype(np.uint16)


def traitement_image(base_path, prefix, suffix, liste_combinaisons, liste_ratios, liste_complexes, acp_choice, acp_combination):
    Logger.info("Processing image...")

    bands, profile = _load_bands(base_path, prefix, suffix)

    if not bands:
        Logger.error("❌ Aucune bande chargée, vérifiez le dossier, le préfixe et le suffixe.")
        return

    # --- Compositions RGB ---
    if liste_combinaisons:
        for name in liste_combinaisons:
            r, g, b = BAND_COMBINATIONS[name]
            if r in bands and g in bands and b in bands:
                rgb_image = np.stack([bands[r], bands[g], bands[b]], axis=0)
                rgb_image = _normalize(rgb_image)
                profile_rgb = profile.copy()
                profile_rgb.update(count=3, dtype=rasterio.uint16, nodata=0)
                output_file = file_output(
                    key=f"Output_{name}",
                    value=f"{prefix}_{suffix}_{name}.tif",
                    label=f"Combinaison RGB {name}",
                    make_path=True,
                )
                with rasterio.open(output_file, "w", **profile_rgb) as dst:
                    dst.write(rgb_image)
                Logger.info(f"✅ Composition {name} générée")
            else:
                Logger.warning(f"❌ Bande manquante pour {name}")

    # --- Ratios de bandes ---
    if liste_ratios:
        profile_ratio = profile.copy()
        profile_ratio.update(count=1, dtype=rasterio.float32, nodata=np.nan)

        for name in liste_ratios:
            num, den = BAND_RATIOS[name]
            if num in bands and den in bands:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(bands[den] != 0, bands[num] / bands[den], np.nan)
                ratio = _normalize(ratio[np.newaxis, ...])[0]
                output_file = file_output(
                    key=f"Output_{name}",
                    value=f"{prefix}_{suffix}_{name}.tif",
                    label=f"Ratio {name}",
                    make_path=True,
                )
                with rasterio.open(output_file, "w", **profile_ratio) as dst:
                    dst.write(ratio.astype(np.float32), 1)
                Logger.info(f"✅ Ratio {name} généré")
            else:
                Logger.warning(f"❌ Bande manquante pour le ratio {name}")

    # --- Calculs algébriques de bandes ---
    if liste_complexes:
        profile_complexes = profile.copy()
        profile_complexes.update(count=1, dtype=rasterio.float32, nodata=np.nan)

        for name in liste_complexes:
            b1, b2, b3, b4, formula = BAND_COMPLEXES[name]
            if all(b in bands for b in [b1, b2, b3, b4]):
                with np.errstate(divide="ignore", invalid="ignore"):
                    complexe = eval(formula)
                complexe = _normalize(complexe[np.newaxis, ...])[0]
                output_file = file_output(
                    key=f"Output_{name}",
                    value=f"{prefix}_{suffix}_{name}.tif",
                    label=f"Calcul algebrique {name}",
                    make_path=True,
                )
                with rasterio.open(output_file, "w", **profile_complexes) as dst:
                    dst.write(complexe.astype(np.float32), 1)
                Logger.info(f"✅ Calcul algébrique {name} généré")
            else:
                Logger.warning(f"❌ Bande manquante pour {name}")

    # --- Calcul l'ACP des bandes ---

    if acp_choice:

        # --- On enlève la bande 1 de Landsat 8 (bruits atmosphériques) ---
      
        bands_for_pca = [b for b in sorted(bands.keys()) if b != 1]
        Logger.info(f"Bandes utilisées pour l'ACP : {bands_for_pca}")

        # --- Empilement en matrice (pixels x bandes) ---
        stack = np.array([bands[i] for i in bands_for_pca])
        n_bands, rows, cols = stack.shape
        X = stack.reshape(n_bands, -1).T

        # --- Masque :  pixels sans NaN dans toutes les bandes ---
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        # --- ACP (échantillonage pour accélerer le calcul) ---
        n_samples = min(500000, X_valid.shape[0])                    # 500 000 échantillons max
        sample_indices = np.random.choice(X_valid.shape[0], size=n_samples, replace=False)
        pca = PCA()
        pca.fit(X_valid[sample_indices])
        Logger.info(f"Variance contenue par les composantes principales : {pca.explained_variance_ratio_.round(6)}")

        X_pca = pca.transform(X_valid)                              # (n_valid_pixels, n_components)

        # --- Reconstruction des images CP en 2D ---

        list_idx = [BAND_ACP[name] for name in acp_choice]

        pc_images = []
        for pc_idx in list_idx:
            pc_img = np.full(rows * cols, np.nan, dtype=np.float32)
            pc_img[valid_mask] = X_pca[:, pc_idx - 1]
            pc_images.append(pc_img.reshape(rows,cols))
        
        # --- Normalisation et assignation RGB ---
        rgb_pca = np.zeros((3, rows, cols), dtype=np.uint16)

        # --- Traduction de la demande de l'utilisateur en une liste de chiffres directement utilisables --- 
        numeros_pc = [int(x) for x in re.findall(r"\d+", acp_combination)]  # traduction de la demande de l'utilisateur en une liste de chiffres directement utilisables
        rgb_mapping = [num -1 for num in numeros_pc]

        for pos_pc, pc_idx in enumerate(rgb_mapping):
            pc_image = pc_images[pc_idx]
            vmin, vmax = np.nanpercentile(pc_image, 2), np.nanpercentile(pc_image, 98)

            if vmax - vmin == 0 : 
                rgb_pca[pos_pc] = 0
            else:
                rgb_pca[pos_pc] = np.clip((pc_image - vmin) / (vmax - vmin) * 65535, 0, 65535).astype(np.uint16)
        
        profile_pca = profile.copy()
        profile_pca.update(count=3, dtype=rasterio.uint16, nodata=0)
        
        output_file = file_output(
            key=f"Output_ACP", 
            value=f"{prefix}_{suffix}_ACP_CP{rgb_mapping[0]}CP{rgb_mapping[1]}CP{rgb_mapping[2]}.tif",
            label=f"Composantes principales",
            make_path=True,
        )
        with rasterio.open(output_file, "w", **profile_pca) as dst:
            dst.write(rgb_pca)
        
        Logger.info(f"✅ Composantes principales générées")



    Logger.info("🎉 Traitement terminé !")
