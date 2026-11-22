# Jellify

Turn the main subject of a photo into wobbly jelly, entirely in the browser.
Drag it with the mouse or a finger, shake the browser window, or shake your phone.

Or switch to **Googly eyes** (the toggle in the header): every face in the
photo gets a pair of googly eyes whose pupils rattle around when you shake the
window or the phone, or drag the eyes. Tap to add an eye, double-tap one to
remove it. **Both** sticks the googly eyes onto the jelly, so they rattle as
it wobbles.

Sound effects (a boing for the jelly, a plastic tick for the pupils) are
synthesised with Web Audio and are off until switched on in Settings.

## Running

```sh
nvm use            # Node 26
npm install
npm run dev        # http://localhost:5173
npm run dev:phone  # HTTPS on your LAN, for testing device motion on a phone
npm run build      # static site in dist/
```

The models are downloaded into `public/models/` by `npm run fetch-model`,
which runs automatically before `dev` and `build`.

On phones, shake detection only works over HTTPS (`dev:phone` uses a
self-signed certificate; accept the warning). iOS only allows motion access
from inside a tap, so the first tap on the photo shows its permission prompt.

## How it works

1. **Segmentation** (`src/segment.js`): [U²-Netp](https://github.com/xuebinqin/U-2-Net),
   a 4.6 MB salient-object model, run with onnxruntime-web's WASM backend.
   It finds the main subject automatically, with no clicks needed. The runtime
   `.wasm` adds about 3.7 MB gzipped, so a first visit downloads about 8 MB.
   Inference runs in a Web Worker (`src/ort-worker.js`) so the jelly keeps
   animating meanwhile.
2. **Mask clean-up** (`src/mask.js`): keep the main blobs, upsample, snap to
   image edges with a guided filter. The inpainting hole is that matte grown
   by a couple of pixels, so the subject's fringe isn't left behind.
3. **Background fill** (`src/inpaint.js`): push-pull pyramid interpolation,
   with no model. Smooth rather than textured.
4. **Jelly** (`src/jelly.js`): a grid mesh over the subject, simulated as a
   mass-spring network. Each vertex is anchored to its rest position, with a
   per-vertex firmness that comes from one of two modes (Settings → "Wobbliest
   part"):
   - **Top of the subject** (default): firm at the base, loose at the top.
   - **Nearest the camera**: firmness from a depth map estimated by
     [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small)
     (int8, 27 MB, `src/depth.js`). It is downloaded only when this mode is
     first chosen, and depth is estimated once per photo. Parts nearest the
     camera wobble most. "Show depth map" draws the scene with the estimated
     depth as an inferno colour map (bright = near). The jelly wobbles in that
     view too, and the background's hole is filled like the photo's.

   In both modes, parts of the subject that touch the photo's edge are pinned
   in place, so they don't pull away from the frame. Edge, shear and bend springs connect
   neighbours, and per-triangle area preservation stops the mesh from folding.
   The wobbliness slider scales all stiffnesses exponentially (4× to ½×), and
   damping scales with stiffness so firm jelly settles quickly.
5. **Rendering** (`src/renderer.js`): WebGL2, with the background as a quad
   and the textured jelly mesh on top. Stretched areas get a slight sheen.
6. **Input** (`src/input.js`): pointer events; `screenX/screenY` polling for
   window shakes (there's no event for that); `devicemotion` for phone shakes.

## Googly eyes

1. **Face detection** (`src/faces.js`): [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   (OpenCV Zoo, MIT, 230 KB), downloaded when the mode is first chosen. It
   is trained on mostly upright faces, and on tilted ones its landmarks drift
   back towards level, so it runs on the photo rotated in 30° steps (12
   passes, about 25 ms each) and keeps each face from the pass that scored
   it highest, where it was nearest upright. That gives each face's eye
   landmarks at any orientation, upside down included; each eye is centred on one, sized to
   0.45× the gap between them, or to the face's width when the head is
   turned and the eyes look close together. They're spread apart along the
   line between the eyes, so they follow the head's tilt. Up to 40 faces get
   eyes.
2. **No faces**: a pair goes near the top of the main subject found by U²-Netp,
   or in the middle of the photo. Eyes can be dragged anywhere, added with a
   tap and removed with a double-tap; "Reset eyes" undoes all of that.
3. **Physics** (`src/googly.js`): each pupil is a disc loose inside its eye,
   simulated in the eye's frame. It falls under gravity, and the eye's own
   acceleration (from window moves, drags, or device motion) pushes it the
   other way. It bounces off the rim and rolls to a stop. Accelerations are
   scaled as if every eye were 5 cm across, so big and small eyes behave alike.
4. **Rendering**: a 2D canvas laid over the WebGL photo.
5. **Both**: each eye is tied to the jelly-mesh triangle under it at rest
   (or the nearest one, if it's just outside), and follows that triangle as
   the jelly deforms. The pupils feel the jelly's movement as well as the
   window's and the phone's. Eyes off the jelly stay put with the background.

## Known limitations

- The background fill is a blur. A small inpainting model (e.g. MI-GAN) would
  be the next upgrade.
- U²-Netp picks one "salient" thing; there's no way to choose a different
  subject yet. A SlimSAM-77 tap-to-select mode (~14 MB) could be added.
- Googly-eye gravity always points down the screen; it doesn't follow the
  phone's tilt.
- Window-shake detection does nothing on Wayland or when the window is
  maximised, because `screenX/Y` don't change.
