# Backrooms: Escape

A browser horror game built with Three.js. Collect his four gadgets, find the exit, and survive.

## Play locally
Open `index.html` in a desktop browser (Chrome or Firefox). It loads Three.js from a CDN, so you need internet access.

## Put it on GitHub Pages
1. Create a new repository and add `index.html` and this README.
2. Go to **Settings → Pages**, choose **Deploy from a branch**, pick `main` and `/ (root)`, then save.
3. Your game will be live at `https://<your-username>.github.io/<repo-name>/`.

## Controls
WASD move · Shift sprint · Mouse look · 1 Chore list · 2 Homework · 3 Money · 4 Bath water

## Notes
- The monster's face photo is chosen on the start screen. It stays in the player's browser and is never uploaded.
- Next steps: swap the blocky monster for a Blender model (export `.glb`, load with `GLTFLoader`) and keep the face mesh's texture linked to `faceTex` in `index.html`.
