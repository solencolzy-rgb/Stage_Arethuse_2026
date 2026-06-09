
# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('test.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

import os
import onecode
import numpy as np
import rasterio
from onecode import Logger, Mode, Project, dropdown, file_input, text_input, file_output
from rasterio.enums import Resampling
from sklearn.decomposition import PCA


# --- Mapping des options dropdown vers les tuples de bandes ---
BAND_COMBINATIONS = {
    "Natural Color (4-3-2)":          (4, 3, 2),
    "False Color (5-4-3)":            (5, 4, 3),
    "Geological structure (7-5-3)":   (7, 5, 3),
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
    'Ferric Iron 4/2x(4+6)/5':                    (2, 4, 6, 5, 'np.where((bands[5] != 0) & (bands[2] != 0), (bands[4]/bands[2])*(bands[4]+bands[6])/bands[5], np.nan)'),
    'Ferrous Iron (3+6)/(4+5)':             (3, 6, 4, 5, 'np.where((bands[4]+bands[5]) != 0, (bands[3]+bands[6])/(bands[4]+bands[5]), np.nan)'),
    'Iron Sulfate 2/1-5/4':             (2, 1, 5, 4, 'np.where((bands[4] != 0) & (bands[1] != 0), (bands[2]/bands[1])-(bands[5]/bands[4]), np.nan)'),
    'Clay Sulfate Mica Marble 6/7-5/4': (6, 7, 5, 4, 'np.where((bands[7] != 0) & (bands[4] != 0), (bands[6]/bands[7])-(bands[5]/bands[4]), np.nan)'),
    'Hydrated Minerals (5-6)/(6+5)':        (5, 6, 6, 5, 'np.where((bands[6]+bands[5]) != 0, (bands[5]-bands[6])/(bands[6]+bands[5]), np.nan)'),
    'Clay Alteration Minerals (6-7)/(6+7)': (6, 7, 6, 7, 'np.where((bands[6]+bands[7]) != 0, (bands[6]-bands[7])/(bands[6]+bands[7]), np.nan)'),
    'Litho Discrimination (6-2)/(6+2)':     (6, 2, 6, 2, 'np.where((bands[6]+bands[2]) != 0, (bands[6]-bands[2])/(bands[6]+bands[2]), np.nan)'),
    'Alteration Minerals (6-5)/(6+5)':      (6, 5, 6, 5, 'np.where((bands[6]+bands[5]) != 0, (bands[6]-bands[5])/(bands[6]+bands[5]), np.nan)')
}


def run():
    onecode.Logger.info(f"Hello {text_input('your name', 'OneCoder')}!")

    base_path = file_input(
        key="InputFolder",
        value="/path/to/landsat",
        label="Dossier bandes Landsat",
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
        value=[],
        label="Choisissez les combinaisons de bandes",
        options=list(BAND_COMBINATIONS.keys()),
        multi=True,
    )

    chosen_ratios = dropdown(
        key="Ratios",
        value=[],
        label="Choisissez les indices de ratios",
        options=list(BAND_RATIOS.keys()),
        multi=True,
    )

    chosen_complexes = dropdown(
        key="Calculs lineaires",
        value=[],
        label="Choisissez les calculs lineaires de bandes",
        options=list(BAND_COMPLEXES.keys()),
        multi=True,
    )

    traitement_image(base_path, prefix, suffix, chosen_combinations, chosen_ratios, chosen_complexes)


def _load_bands(base_path, prefix, suffix, band_range=range(1, 8)):
    """Charge les bandes Landsat depuis le dossier."""
    bands = {}
    profile = None
    nodata_value = -9999.0

    for i in band_range:
        file_path = os.path.join(base_path, f"{prefix}_B{i}_{suffix}.tif")
        if os.path.exists(file_path):
            with rasterio.open(file_path) as src:
                band_data = src.read(1, resampling=Resampling.nearest).astype(np.float32)
                band_data[band_data == nodata_value] = np.nan
                bands[i] = band_data
                profile = src.profile
        else:
            Logger.warning(f"⚠️ Fichier manquant : {file_path}")

    return bands, profile


def _normalize(image, per_band=True):
    """Normalisation percentile 2-98."""
    normalized = np.zeros_like(image)
    for idx in range(image.shape[0]):
        band = image[idx]
        vmin, vmax = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
        normalized[idx] = np.clip((band - vmin) / (vmax - vmin) * 65535, 0, 65535)
    return normalized.astype(np.uint16)


def traitement_image(base_path, prefix, suffix, liste_combinaisons, liste_ratios, liste_complexes):
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
                    value=f"{prefix}_{suffix}_{name}.tif",  # relatif au dossier outputs/ automatiquement
                    label=f"Combinaison RGB {name}",
                    make_path=True,                # crée le dossier si il n'existe pas
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
                    ratio = _normalize(ratio[np.newaxis, ...])[0]  # Normalisation et suppression de la dimension inutile
                output_file = file_output(
                        key=f"Output_{name}",
                        value=f"{prefix}_{suffix}_{name}.tif",  # relatif au dossier outputs/ automatiquement
                        label=f"Ratio {name}",
                        make_path=True,                # crée le dossier si il n'existe pas
                        )
                with rasterio.open(output_file, "w", **profile_ratio) as dst:
                    dst.write(ratio.astype(np.float32), 1)
                Logger.info(f"✅ Ratio {name} généré")
            else:
                Logger.warning(f"❌ Bande manquante pour le ratio {name}")
    
    # --- PCA ---

    # --- Calculs algébriques de bandes ---
    if liste_complexes:
        profile_complexes = profile.copy()
        profile_ratio.update(count=1, dtype=rasterio.float32, nodata=np.nan)

        for name in liste_complexes:
            b1, b2, b3, b4, formula = BAND_COMPLEXES[name]
            if all(b in bands for b in [b1, b2, b3, b4]):
                complexe = eval(formula)
                complexe = _normalize(complexe[np.newaxis, ...])[0]  # Normalisation et suppression de la dimension inutile
                output_file = file_output(
                        key=f"Output_{name}",
                        value=f"{prefix}_{suffix}_{name}.tif",  # relatif au dossier outputs/ automatiquement
                        label=f"Calcul algébrique {name}",
                        make_path=True,                # crée le dossier si il n'existe pas
                        )
                with rasterio.open(output_file, "w", **profile_complexes) as dst:
                    dst.write(complexe.astype(np.float32), 1)
                Logger.info(f"✅ Calcul algébrique de bandes {name} généré")
            return 

    Logger.info("🎉 Traitement terminé !")
