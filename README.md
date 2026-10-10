# Jellify

Turn the main subject of a photo into wobbly jelly, entirely in the browser.
Drag it with the mouse or a finger, shake the browser window, or shake your phone.

## Running

```sh
nvm use            # Node 26
npm install
npm run dev        # http://localhost:5173
npm run dev:phone  # HTTPS on your LAN, for testing device motion on a phone
npm run build      # static site in dist/
```

The model is downloaded into `public/models/` by `npm run fetch-model`, which
runs automatically before `dev` and `build`.

On phones, shake detection only works over HTTPS (`dev:phone` uses a
self-signed certificate; accept the warning). iOS also needs the
"Enable shake" button tapped once to grant motion access.

## How it works

1. **Segmentation** (`src/segment.js`): [U²-Netp](https://github.com/xuebinqin/U-2-Net),
   a 4.6 MB salient-object model, run with onnxruntime-web's WASM backend.
   It finds the main subject automatically, with no clicks needed. The runtime
   `.wasm` adds about 3.7 MB gzipped, so a first visit downloads about 8 MB.
2. **Mask clean-up** (`src/mask.js`): keep the main blobs, upsample, snap to
   image edges with a guided filter, then dilate to get the inpainting hole.
3. **Background fill** (`src/inpaint.js`): push-pull pyramid interpolation,
   with no model. Smooth rather than textured.
4. **Jelly** (`src/jelly.js`): a grid mesh over the subject, simulated as a
   mass-spring network. Each vertex is anchored to its rest position (stiff at
   the base, loose at the top). Edge, shear and bend springs connect
   neighbours, and per-triangle area preservation stops the mesh from folding.
   The jelly layer is opaque over the whole hole and feathers out past it, so
   at rest the picture is pixel-identical to the photo.
5. **Rendering** (`src/renderer.js`): WebGL2, with the background as a quad
   and the textured jelly mesh on top. Stretched areas get a slight sheen.
6. **Input** (`src/input.js`): pointer events; `screenX/screenY` polling for
   window shakes (there's no event for that); `devicemotion` for phone shakes.

## Known limitations

- The background fill is a blur. A small inpainting model (e.g. MI-GAN) would
  be the next upgrade.
- U²-Netp picks one "salient" thing; there's no way to choose a different
  subject yet. A SlimSAM-77 tap-to-select mode (~14 MB) could be added.
- Window-shake detection does nothing on Wayland or when the window is
  maximised, because `screenX/Y` don't change.
