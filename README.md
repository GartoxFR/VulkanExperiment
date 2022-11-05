# Contexte
Ce repo contient un projet perso qui m'a servis d'introduction à Vulkan avec le
langage Rust.

# Binaries
## Mandelbrot
```cargo run --bin mandelbrot```

Premier programme écrit pour se familiraiser avec la compute pipeline.

Génère l'ensemble de mandelbrot dans une image en laissant la carte graphique
faire les calculs

## Graphics
```cargo run --bin graphics```

Programme écrit pour se familiariser avec la graphics pipeline. Il dessine
simplement un triangle dans une fenètre.

## Slime
```cargo run --bin slime```

Combine la compute pipeline et la graphics pipeline pour afficher une
simulation de petits agents (pixels) qui suivent des règles simples dans une
image.

Ce projet est très fortement inspiré de l'excellente vidéo de Sebasian Lague :
[Ant and Slime simulations](https://www.youtube.com/watch?v=X-iSQQgOd1A).

