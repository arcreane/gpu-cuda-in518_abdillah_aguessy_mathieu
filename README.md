[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Xzr3Zei0)

# PartiQleStudio

Simulateur de particules 2D haute performance (CPU/GPU – CUDA), développé avec Qt et Raylib.

## Description

PartiQleStudio est un moteur de simulation physique en temps réel permettant :

- Calcul CPU ou GPU (CUDA)
- Gestion complète des collisions, gravité, friction et amortissement
- Interaction souris : attraction, explosion et poussée dynamique
- Presets physiques : Terre, Mars, Vide spatial
- Graphique en temps réel FPS / FrameTime
- Debug overlays configurables (moteur, performances, souris, cadre de simulation)

---

## Fonctionnalités

### Moteur de simulation
- Calcul CPU séquentiel optimisé
- Calcul GPU parallèle (CUDA)
- Jusqu'à plusieurs dizaines de milliers de particules
- Collisions inter-particules + collisions murs
- Intégration Raylib pour le rendu temps réel
- Nombre particules maximum : 100000

### Paramètres physiques ajustables
- Élasticité (0.0 à 1.0)
- Friction visqueuse (0.0 à 1.0)
- Gravité (0 à 300)
- Amortissement global (0.900 à 1.0)
- Rayon min / max des particules (1 - 20)
- Vitesse initiale min / max (0 - 100)
- Rayon et force de la souris (0 - 100 px) et (0 - 5)

### Interaction utilisateur
- Poussée selon mouvement de souris
- Attraction (clic gauche)
- Explosion (clic droit)
- Boutons Start, Pause, Reset
- Presets Terre / Mars / Vide spatial
- Réinitialisation des paramètres physiques

### Visualisation
- Graphique FPS / FrameTime intégré dans Qt
- Debug overlays :
  - Informations moteur
  - Informations souris
  - Performances
  - Cadre de simulation Raylib

---

## Prérequis

- Qt 6.x  
- Raylib 5.x  
- Visual Studio 2022 (C++17)  
- CUDA Toolkit 11.x ou supérieur (optionnel pour mode GPU)  
- CMake 3.20+ (si construction hors Visual Studio)

---

## Installation

### Clonage du dépôt

```bash
git clone https://github.com/arcreane/gpu-cuda-in518_abdillah_aguessy_mathieu.git
cd gpu-cuda-in518_abdillah_aguessy_mathieu
```

## Configuration

1. Ouvrir `PartiQleStudio.sln` dans Visual Studio 2022
2. Vérifier les chemins vers Qt et Raylib dans les propriétés du projet
3. Pour activer CUDA : définir `USE_CUDA` dans les définitions du préprocesseur

### Compilation

**Mode CPU uniquement :**
```bash
Build > Build Solution (Ctrl+Shift+B)
```

**Mode CPU + GPU (CUDA) :**
1. Installer le CUDA Toolkit
2. Vérifier que `kernel.cu` est bien compilé via NVCC
3. Build en configuration Release pour performances optimales

## Utilisation

### Démarrage d'une simulation

1. **Configurer le nombre de particules** (spinbox)
2. **Choisir le mode** : CPU ou GPU
3. **Ajuster les paramètres physiques** ou sélectionner un preset
4. **Cliquer sur "Start"**

### Presets Disponibles
`Options > Presets > ...`
- **Terre** : Gravité normale, friction moyenne
- **Mars** : Gravité réduite, friction faible
- **Vide Spatial** : Pas de gravité, pas de friction, haute élasticité

### Overlays Debug
`View > Raylib > ...`

Menu **Affichage** :
- ☑️ Afficher tout
- ☑️ Infos souris
- ☑️ Infos moteur
- ☑️ Infos performances
- ☑️ Infos boîte
- ☑️ Graphique FPS

## Structure du Projet
```css
├── PartiQleStudio/
│   ├── src/
│   │   ├── MainWindow.cpp
│   │   ├── RaylibView.cpp
│   │   ├── FpsGraphWidget.cpp
│   │   └── main.cpp
│   │
│   ├── src/
│   │   ├── MainWindow.h
│   │   ├── RaylibView.h
│   │   ├── FpsGraphWidget.h
│   │   └── cuda_api.h
│   |
│   ├── cuda/
│   │   └── kernel.cu
│   │
│   ├── ui/
│       └── MainWindow.ui
│
├── PartiQleStudio.sln
├── CMakeLists
├── .gitignore
└── README.md
```

## Architecture

### CPU Mode
- Simulation séquentielle
- Gravité, friction, amortissement appliqués à chaque frame
- Collisions inter-particules
- Interaction souris : attraction, explosion, poussée

### GPU Mode (CUDA)
- Allocation mémoire GPU (`cuda_particles_init`)
- Upload CPU → GPU
- Calcul parallèle via `cuda_particles_step`
- Download GPU → CPU pour le rendu Raylib
- Interaction souris via `cuda_apply_mouse_force`

## Performance

### Comparaison des performances CPU / GPU

| Particules | FPS CPU | FPS GPU |
|-----------:|--------:|--------:|
| 1 000      | 60 fps  | 60 fps  | 
| 5 000      | 60 fps  | 60 fps  |
| 10 000     | 20 fps  | 60 fps  |
| 12 000     | 14 fps  | 58 fps  |
| 14 000     | 11 fps  | 55 fps  |
| 20 000     |  5 fps  | 40 fps  |
| 30 000     |  2 fps  | 25 fps  |

### Configuration de la machine de test
*Tests effectués sur un Intel Core i9-12900H, 16 Go RAM, NVIDIA RTX 3070 Ti Laptop GPU.*

CPU :
- **Modèle :** Intel Core i9-12900H  
- **Cœurs :** 14  
- **Threads :** 20  
- **Fréquence Max :** 2.9 GHz  

GPU :
- **GPU principal (CUDA) :** NVIDIA GeForce RTX 3070 Ti Laptop GPU  
- **GPU intégré :** Intel Iris Xe Graphics  
- **Affichage virtuel :** USB Mobile Monitor (sans impact sur CUDA)  

RAM :
- **Capacité totale :** 17 179 869 184 bytes ≈ **16 Go RAM**  
- **Nombre de barrettes :** 2  

## Auteurs

- Abdillah Raïssa
- Aguessy Maëva
- Mathieu Baptiste

Projet réalisé dans le cadre du module IN518 – Calcul Haute Performance GPU (IPSA A5).

## Problèmes Connus

- Le mode GPU nécessite une carte NVIDIA compatible CUDA
- Sur certaines configurations, le graphique FPS peut causer des ralentissements

## Améliorations Futures

- [ ] Support OpenCL pour GPUs AMD
- [ ] Chargement/sauvegarde de presets personnalisés
- [ ] Colorisation des particules selon vélocité/énergie
- [ ] Support multi-GPU

---

**Cours**: IN518 - Haute Performance GPU  
**Institution**: IPSA  
**Année**: 2025-2026
