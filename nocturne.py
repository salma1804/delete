import cv2
import numpy as np

# Charger l'image
image = cv2.imread('image_nuit3.jpg')  # Remplace 'image_nuit.jpg' par le nom de ton image
if image is None:
    print("❌ Image non trouvée !")
    exit()

# Convertir en niveaux de gris (Étape 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Afficher l'image d'origine et l'image en niveaux de gris
cv2.imshow('Image d\'origine', image)
cv2.waitKey(1000)  # Attendre 1 seconde
cv2.imshow('Image Grayscale', gray)
cv2.waitKey(1000)

# Égalisation d’histogramme (Étape 2)
equalized = cv2.equalizeHist(gray)
cv2.imshow('Égalisation d\'histogramme', equalized)
cv2.waitKey(1000)

# CLAHE (amélioration locale du contraste) (Étape 3)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_applied = clahe.apply(gray)
cv2.imshow('CLAHE Appliqué', clahe_applied)
cv2.waitKey(1000)

# Réduction de bruit (Étape 4)
denoised = cv2.fastNlMeansDenoising(clahe_applied, h=10)
cv2.imshow('Image Denoisée', denoised)
cv2.waitKey(1000)

# Légère augmentation de la luminosité (Étape 5)
bright_plus = cv2.convertScaleAbs(denoised, alpha=1.1, beta=20)
cv2.imshow('Image avec Luminosité Augmentée', bright_plus)
cv2.waitKey(1000)

# Seuillage d’Otsu sur l'image améliorée (Étape 6)
_, otsu = cv2.threshold(bright_plus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Seuillage d\'Otsu', otsu)
cv2.waitKey(1000)

# Ajouter du texte sur l'image résultante
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(bright_plus, 
            'Amélioration de la visibilité nocturne', 
            (50, 50), 
            font, 
            1, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA)

# Afficher l'image finale avec texte
cv2.imshow('Image Résultante', bright_plus)
cv2.waitKey(0)  # Attendre jusqu'à ce que l'utilisateur ferme la fenêtre

# Sauvegarder l'image avec texte
cv2.imwrite('resultat_avec_texte.jpg', bright_plus)

# Fermer toutes les fenêtres ouvertes
cv2.destroyAllWindows()
