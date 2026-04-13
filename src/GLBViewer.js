/**
 * GLB avatar viewer with rotation control (0–360°).
 * Uses WebView + Three.js to load and display a GLB; rotation is driven from parent via injectJavaScript.
 * Prepared for clothing: Option A = 3D garments (gltf) skinned to avatar rig; Option B = cloth sim / size later.
 */
import React, { useRef, useEffect, useState } from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { WebView } from 'react-native-webview';

export const VISIBILITY_MODES = {
  AUTO: 'AUTO',
};

const makeHTML = (glbUrl, neutralLighting) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, user-scalable=no">
  <style>body { margin: 0; background: #23262b; overflow: hidden; }</style>
</head>
<body>
  <script>
    var scene, camera, renderer, mesh, mixer, clock;
    var rotationYDeg = 0;
    var glbUrl = ${JSON.stringify(glbUrl)};
    var neutralLighting = ${neutralLighting ? 'true' : 'false'};
    var touchStartX, touchStartDeg;
    var THREE_CDN = "https://unpkg.com/three@0.128.0/build/three.min.js";
    var GLTFLOADER_CDN = "https://unpkg.com/three@0.128.0/examples/js/loaders/GLTFLoader.js";
    var ROOMENV_CDN = "https://unpkg.com/three@0.128.0/examples/js/environments/RoomEnvironment.js";
    var RGBELOADER_CDN = "https://unpkg.com/three@0.128.0/examples/js/loaders/RGBELoader.js";
    // Small-ish 1K studio HDRI (fallback to RoomEnvironment if it fails).
    var HDRI_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_08_1k.hdr";
    var TARGET_HUMAN_HEIGHT_M = 1.7;

    function post(type, payload) {
      try {
        if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: type, ...(payload || {}) }));
        }
      } catch (_) {}
    }

    window.onerror = function(message, source, lineno, colno, error) {
      post('error', {
        stage: 'window.onerror',
        message: String(message || 'Ukjent JS-feil'),
        source: source ? String(source) : null,
        lineno: typeof lineno === 'number' ? lineno : null,
        colno: typeof colno === 'number' ? colno : null,
        name: error && error.name ? String(error.name) : null,
        stack: error && error.stack ? String(error.stack) : null
      });
    };
    window.onunhandledrejection = function(ev) {
      var r = ev && ev.reason ? ev.reason : null;
      post('error', {
        stage: 'unhandledrejection',
        message: r ? String(r.message || r) : 'Ukjent promise-feil',
        name: r && r.name ? String(r.name) : null,
        stack: r && r.stack ? String(r.stack) : null
      });
    };

    function loadScript(src) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = src;
        s.async = true;
        s.onload = function() { resolve(true); };
        s.onerror = function(e) { reject(new Error('Script load failed: ' + src)); };
        document.head.appendChild(s);
      });
    }

    function init() {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x23262b);
      camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
      camera.position.set(0, 1, 2.5);
      camera.lookAt(0, 0.8, 0);
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.5));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      // Premium look: physically correct lights + ACES tonemapping.
      renderer.physicallyCorrectLights = true;
      if (renderer.outputColorSpace !== undefined && THREE.SRGBColorSpace !== undefined) {
        renderer.outputColorSpace = THREE.SRGBColorSpace;
      } else if (THREE.sRGBEncoding) {
        renderer.outputEncoding = THREE.sRGBEncoding;
      }
      if (renderer.toneMapping !== undefined) {
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
      }
      document.body.appendChild(renderer.domElement);
      function applyRoomEnvironmentFallback() {
        try {
          if (THREE.PMREMGenerator && THREE.RoomEnvironment) {
            var pmremFallback = new THREE.PMREMGenerator(renderer);
            var envFallback = pmremFallback.fromScene(new THREE.RoomEnvironment(), 0.04).texture;
            scene.environment = envFallback;
          }
        } catch (_) {}
      }

      function setupStudioLighting() {
        // Subtle ambient via environment + 3-point soft lights for nice catchlights.
        // Slightly stronger fill to fake softer "SSS-like" skin (less harsh contrast).
        var hemi = new THREE.HemisphereLight(0xffffff, 0x3a3f46, 0.28);
        hemi.position.set(0, 2.0, 0);
        scene.add(hemi);

        // Softer key (avoid plastic-y hard spec).
        var key = new THREE.DirectionalLight(0xffffff, 2.2);
        key.position.set(2.2, 2.9, 2.0);
        key.target.position.set(0, 0.9, 0);
        key.castShadow = true;
        key.shadow.mapSize.width = 2048;
        key.shadow.mapSize.height = 2048;
        key.shadow.radius = 8;
        key.shadow.bias = -0.00015;
        key.shadow.camera.left = -2.2;
        key.shadow.camera.right = 2.2;
        key.shadow.camera.top = 2.2;
        key.shadow.camera.bottom = -2.2;
        key.shadow.camera.near = 0.2;
        key.shadow.camera.far = 14;
        scene.add(key.target);
        scene.add(key);

        var fill = new THREE.DirectionalLight(0xffffff, 1.0);
        fill.position.set(-2.6, 2.2, 2.5);
        fill.target.position.set(0, 0.9, 0);
        scene.add(fill.target);
        scene.add(fill);

        var rim = new THREE.DirectionalLight(0xffffff, 0.95);
        rim.position.set(0.0, 2.4, -3.2);
        rim.target.position.set(0, 0.95, 0);
        scene.add(rim.target);
        scene.add(rim);

        // Tiny catchlight near camera to bring eyes to life.
        var catchLight = new THREE.PointLight(0xffffff, 0.35, 10.0);
        catchLight.position.set(0.4, 1.45, 2.2);
        scene.add(catchLight);
      }

      // Environment map (HDRI) for realistic reflections.
      (function initEnvAndLights() {
        setupStudioLighting();
        if (!THREE.RGBELoader || !THREE.PMREMGenerator) {
          applyRoomEnvironmentFallback();
          return;
        }
        try {
          var pmrem = new THREE.PMREMGenerator(renderer);
          pmrem.compileEquirectangularShader();
          new THREE.RGBELoader().setDataType(THREE.UnsignedByteType).load(
            HDRI_URL,
            function(hdrTexture) {
              var envMap = pmrem.fromEquirectangular(hdrTexture).texture;
              scene.environment = envMap;
              hdrTexture.dispose && hdrTexture.dispose();
              pmrem.dispose && pmrem.dispose();
              post('status', { stage: 'env', message: 'HDRI lastet' });
            },
            undefined,
            function() {
              applyRoomEnvironmentFallback();
              post('status', { stage: 'env', message: 'HDRI feilet, bruker fallback' });
            }
          );
        } catch (_) {
          applyRoomEnvironmentFallback();
        }
      })();

      var groundGeo = new THREE.PlaneGeometry(6, 6);
      var groundMat = new THREE.ShadowMaterial({ opacity: 0.22 });
      var ground = new THREE.Mesh(groundGeo, groundMat);
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -0.95;
      ground.receiveShadow = true;
      scene.add(ground);

      var root = new THREE.Group();
      root.name = 'avatarRoot';
      root.rotation.order = 'YXZ';
      scene.add(root);

      function normalizeModelOrientation(rootGroup) {
        if (rootGroup.userData && rootGroup.userData.orientationNormalized) return rootGroup.userData.orientationDebug || {};
        var box = new THREE.Box3().setFromObject(rootGroup);
        if (box.isEmpty && box.isEmpty()) return;
        var center = new THREE.Vector3();
        var size = new THREE.Vector3();
        box.getCenter(center);
        box.getSize(size);
        if (!isFinite(size.x) || !isFinite(size.y) || !isFinite(size.z)) return;
        if (size.x < 1e-6 && size.y < 1e-6 && size.z < 1e-6) return;

        var bboxBefore = { x: size.x, y: size.y, z: size.z };
        var heightAxisDetected = (size.x >= size.y && size.x >= size.z) ? 'X' : (size.z >= size.y ? 'Z' : 'Y');
        var rootTransformBefore = {
          position: [rootGroup.position.x, rootGroup.position.y, rootGroup.position.z],
          rotation: [rootGroup.rotation.x, rootGroup.rotation.y, rootGroup.rotation.z],
          scale: [rootGroup.scale.x, rootGroup.scale.y, rootGroup.scale.z],
        };
        var orientationFixApplied = false;
        var scaleFixApplied = false;
        rootGroup.rotation.set(0, 0, 0);

        function evalCandidate(axis, rad) {
          rootGroup.rotation.set(0, 0, 0);
          if (axis === 'X') rootGroup.rotation.x = rad;
          if (axis === 'Z') rootGroup.rotation.z = rad;
          var b = new THREE.Box3().setFromObject(rootGroup);
          var s = new THREE.Vector3();
          b.getSize(s);
          var yLargest = (s.y >= s.x && s.y >= s.z);
          var score = s.y - 0.20 * (s.x + s.z) + (yLargest ? s.y : 0.0);
          return { axis: axis, rad: rad, score: score, size: { x: s.x, y: s.y, z: s.z } };
        }

        var candidateResults = [{ axis: 'NONE', rad: 0.0, score: -Infinity, size: bboxBefore }];
        if (heightAxisDetected === 'X') {
          candidateResults = [evalCandidate('Z', Math.PI / 2), evalCandidate('Z', -Math.PI / 2)];
        } else if (heightAxisDetected === 'Z') {
          candidateResults = [evalCandidate('X', Math.PI / 2), evalCandidate('X', -Math.PI / 2)];
        } else {
          candidateResults = [evalCandidate('NONE', 0.0)];
        }
        var best = candidateResults[0];
        for (var i = 1; i < candidateResults.length; i += 1) {
          if (candidateResults[i].score > best.score) best = candidateResults[i];
        }
        rootGroup.rotation.set(0, 0, 0);
        if (best.axis === 'X') rootGroup.rotation.x = best.rad;
        if (best.axis === 'Z') rootGroup.rotation.z = best.rad;
        var appliedRotAxis = best.axis;
        var appliedRotRad = best.rad;
        orientationFixApplied = Math.abs(appliedRotRad) > 1e-6;

        box.setFromObject(rootGroup);
        box.getSize(size);
        var bboxAfterRotation = { x: size.x, y: size.y, z: size.z };

        // 4) Uniform scale normalization to target human height.
        var bboxHeightY = Math.max(size.y, 1e-6);
        var appliedScaleFactor = TARGET_HUMAN_HEIGHT_M / bboxHeightY;
        if (isFinite(appliedScaleFactor) && appliedScaleFactor > 0) {
          rootGroup.scale.set(appliedScaleFactor, appliedScaleFactor, appliedScaleFactor);
          scaleFixApplied = Math.abs(appliedScaleFactor - 1.0) > 1e-3;
        } else {
          appliedScaleFactor = 1.0;
        }

        // 5) Ground model and center in X/Z.
        box.setFromObject(rootGroup);
        box.getCenter(center);
        var groundOffsetApplied = -box.min.y;
        rootGroup.position.set(-center.x, groundOffsetApplied, -center.z);
        box.setFromObject(rootGroup);
        box.getSize(size);
        var bboxAfter = { x: size.x, y: size.y, z: size.z };
        var rootTransformAfter = {
          position: [rootGroup.position.x, rootGroup.position.y, rootGroup.position.z],
          rotation: [rootGroup.rotation.x, rootGroup.rotation.y, rootGroup.rotation.z],
          scale: [rootGroup.scale.x, rootGroup.scale.y, rootGroup.scale.z],
        };
        var out = {
          bboxBefore: bboxBefore,
          bboxAfter: bboxAfter,
          rootTransformBefore: rootTransformBefore,
          rootTransformAfter: rootTransformAfter,
          orientationFixApplied: orientationFixApplied,
          scaleFixApplied: scaleFixApplied,
          appliedRotationFix: orientationFixApplied,
          appliedScaleFix: scaleFixApplied,
          heightAxisDetected: heightAxisDetected,
          appliedRotAxis: appliedRotAxis,
          appliedRotRad: appliedRotRad,
          appliedRotationRadians: appliedRotRad,
          bboxAfterRotation: bboxAfterRotation,
          detectedUpAxis: heightAxisDetected,
          appliedScaleFactor: appliedScaleFactor,
          groundOffsetApplied: groundOffsetApplied,
          orientationNormalized: true,
        };
        if (!rootGroup.userData) rootGroup.userData = {};
        rootGroup.userData.baseRotX = (appliedRotAxis === 'X') ? appliedRotRad : 0.0;
        rootGroup.userData.baseRotZ = (appliedRotAxis === 'Z') ? appliedRotRad : 0.0;
        rootGroup.userData.orientationNormalized = true;
        rootGroup.userData.orientationDebug = out;
        return out;
      }

      console.log('Rendering avatar from:', glbUrl);
      var loader = new THREE.GLTFLoader();
      // Best-effort: preflight to surface ATS/CORS/404 issues clearly.
      try {
        fetch(glbUrl, { method: 'HEAD' }).then(function(res) {
          post('net', { stage: 'head', url: glbUrl, ok: !!res.ok, status: res.status, statusText: res.statusText });
        }).catch(function(err) {
          post('net', { stage: 'head', url: glbUrl, ok: false, message: String(err && err.message ? err.message : err) });
        });
      } catch (_) {}
      loader.load(glbUrl, function(gltf) {
        mesh = gltf.scene;
        var meshCount = 0;
        var matCount = 0;
        var transparentCount = 0;
        var minOpacity = 1.0;
        var maxOpacity = 0.0;

        function isLikelySkinMaterialName(name) {
          var s = (name || '').toLowerCase();
          return s.indexOf('skin') !== -1 || s.indexOf('body') !== -1 || s.indexOf('face') !== -1 || s.indexOf('head') !== -1;
        }

        function isLikelyEyeName(name) {
          var s = (name || '').toLowerCase();
          return (
            s.indexOf('eye') !== -1 ||
            s.indexOf('iris') !== -1 ||
            s.indexOf('pupil') !== -1 ||
            s.indexOf('sclera') !== -1 ||
            s.indexOf('cornea') !== -1 ||
            s.indexOf('eyeball') !== -1
          );
        }

        function isLikelyCorneaName(name) {
          var s = (name || '').toLowerCase();
          return s.indexOf('cornea') !== -1 || s.indexOf('eye_shell') !== -1 || s.indexOf('eyeshell') !== -1 || s.indexOf('eye_wet') !== -1;
        }

        function setTextureEncoding(tex, wantSRGB) {
          if (!tex) return;
          // r128 uses .encoding; newer uses .colorSpace (handled elsewhere already).
          if (tex.colorSpace !== undefined && THREE.SRGBColorSpace !== undefined) {
            var linearCS = (THREE.LinearSRGBColorSpace !== undefined) ? THREE.LinearSRGBColorSpace : ((THREE.NoColorSpace !== undefined) ? THREE.NoColorSpace : null);
            tex.colorSpace = wantSRGB ? THREE.SRGBColorSpace : (linearCS || tex.colorSpace);
          } else if (tex.encoding !== undefined) {
            tex.encoding = wantSRGB ? THREE.sRGBEncoding : THREE.LinearEncoding;
          }
        }

        function tuneTexture(tex) {
          if (!tex) return;
          // Filtering & mipmaps for sharper, less shimmery output.
          if (tex.generateMipmaps !== undefined) tex.generateMipmaps = true;
          if (tex.minFilter !== undefined) tex.minFilter = THREE.LinearMipmapLinearFilter;
          if (tex.magFilter !== undefined) tex.magFilter = THREE.LinearFilter;
          if (tex.anisotropy !== undefined && renderer && renderer.capabilities && renderer.capabilities.getMaxAnisotropy) {
            tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
          }
          tex.needsUpdate = true;
        }

        function tuneMaterialTextures(m) {
          if (!m) return;
          // Color space correctness.
          setTextureEncoding(m.map, true);
          setTextureEncoding(m.emissiveMap, true);
          setTextureEncoding(m.normalMap, false);
          setTextureEncoding(m.roughnessMap, false);
          setTextureEncoding(m.metalnessMap, false);
          setTextureEncoding(m.aoMap, false);
          setTextureEncoding(m.alphaMap, false);
          setTextureEncoding(m.ormMap, false);

          // Filtering.
          tuneTexture(m.map);
          tuneTexture(m.normalMap);
          tuneTexture(m.roughnessMap);
          tuneTexture(m.metalnessMap);
          tuneTexture(m.aoMap);
          tuneTexture(m.emissiveMap);
          tuneTexture(m.alphaMap);
        }

        function applySkinTuning(m) {
          if (!m) return;
          if (typeof m.metalness === 'number') m.metalness = 0.0;
          // Baseline: soft-ish skin, avoid harsh spec.
          if (typeof m.roughness === 'number') m.roughness = Math.max(0.70, Math.min(0.85, m.roughness || 0.78));
          if (m.roughnessMap) {
            // Ensure roughness map contributes; don't flip unless we have explicit info.
            m.roughness = Math.max(0.70, Math.min(0.90, m.roughness || 0.78));
          }
          if (m.normalMap && m.normalScale && m.normalScale.set) {
            // Subtle normal detail for skin.
            m.normalScale.set(0.5, 0.5);
          }
          // If we lack normals, use AO a bit more to recover depth.
          if (m.aoMap && m.aoMapIntensity !== undefined) {
            m.aoMapIntensity = m.normalMap ? 0.45 : 0.70;
          }
          if (m.envMapIntensity !== undefined) {
            // Lower reflections on skin for less "plastic" look.
            m.envMapIntensity = 0.70;
          }
          // Reduce "clearcoat/plastic" if present.
          if (m.clearcoat !== undefined) m.clearcoat = 0.0;
          if (m.clearcoatRoughness !== undefined) m.clearcoatRoughness = 1.0;
          if (m.reflectivity !== undefined) m.reflectivity = 0.15;
          // Fake SSS: gentle warm lift in shadows via tiny emissive.
          if (m.emissive && m.emissive.setRGB && !m.emissiveMap) {
            m.emissive.setRGB(0.10, 0.03, 0.025);
            if (m.emissiveIntensity !== undefined) m.emissiveIntensity = 0.035;
          }
          if (m.dithering !== undefined) m.dithering = true;
          m.needsUpdate = true;
        }

        function looksEyeByTexture(m) {
          if (!m) return false;
          var candidates = [m.map, m.emissiveMap, m.normalMap, m.roughnessMap, m.metalnessMap, m.aoMap].filter(Boolean);
          for (var i = 0; i < candidates.length; i += 1) {
            var t = candidates[i];
            var n = (t && t.name) ? String(t.name).toLowerCase() : '';
            var src = '';
            try { src = (t && t.image && t.image.src) ? String(t.image.src).toLowerCase() : ''; } catch (_) {}
            var s = n + ' ' + src;
            if (s.indexOf('eye') !== -1 || s.indexOf('iris') !== -1 || s.indexOf('sclera') !== -1 || s.indexOf('cornea') !== -1 || s.indexOf('pupil') !== -1) return true;
          }
          return false;
        }

        function applyEyeTuning(meshNode, m) {
          if (!m) return m;
          var nodeName = (meshNode && meshNode.name) ? String(meshNode.name) : '';
          var matName = m && m.name ? String(m.name) : '';
          var wantsCornea = isLikelyCorneaName(nodeName) || isLikelyCorneaName(matName);

          // If eye parts share skin material, clone so we can tune without affecting skin.
          var tuned = m;
          if (isLikelySkinMaterialName(matName)) {
            tuned = m.clone();
            tuned.name = (m.name ? m.name : 'mat') + '_eyeOverride';
          }

          if (typeof tuned.metalness === 'number') tuned.metalness = 0.0;
          if (tuned.envMapIntensity !== undefined) tuned.envMapIntensity = 1.1;

          if (wantsCornea) {
            // Cornea / wet shell: glossy highlight.
            if (tuned.clearcoat !== undefined) tuned.clearcoat = 1.0;
            if (tuned.clearcoatRoughness !== undefined) tuned.clearcoatRoughness = 0.03;
            if (typeof tuned.roughness === 'number') tuned.roughness = 0.14;
            if (tuned.envMapIntensity !== undefined) tuned.envMapIntensity = 1.25;
          } else {
            // Iris/sclera: keep some gloss, but not mirror-like.
            if (typeof tuned.roughness === 'number') tuned.roughness = Math.max(0.40, Math.min(0.60, tuned.roughness || 0.50));
          }

          // Slightly off-white sclera if baseColorFactor exists (avoid pure white).
          if (tuned.color && tuned.color.setRGB && (nodeName.toLowerCase().indexOf('sclera') !== -1 || matName.toLowerCase().indexOf('sclera') !== -1)) {
            tuned.color.setRGB(0.94, 0.94, 0.93);
          }

          tuned.needsUpdate = true;
          return tuned;
        }

        function buildTextureDebugList(rootScene) {
          var texSet = [];
          var seen = new Set();
          function push(tex, usage, owner) {
            if (!tex) return;
            if (seen.has(tex.uuid)) return;
            seen.add(tex.uuid);
            var img = tex.image || null;
            var w = img && img.width ? img.width : null;
            var h = img && img.height ? img.height : null;
            var cs = null;
            if (tex.colorSpace !== undefined && tex.colorSpace !== null) cs = String(tex.colorSpace);
            else if (tex.encoding !== undefined && tex.encoding !== null) cs = String(tex.encoding);
            texSet.push({
              name: tex.name || null,
              usage: usage || null,
              owner: owner || null,
              width: w,
              height: h,
              colorSpace: cs,
              flipY: tex.flipY === true,
              anisotropy: tex.anisotropy || 0,
              minFilter: tex.minFilter,
              magFilter: tex.magFilter,
              mipmaps: tex.generateMipmaps === true
            });
          }
          rootScene.traverse(function(n) {
            if (!(n.isMesh || n.isSkinnedMesh) || !n.material) return;
            var mats = Array.isArray(n.material) ? n.material : [n.material];
            mats.forEach(function(m) {
              if (!m) return;
              var owner = (n.name || '') + ' / ' + (m.name || '');
              push(m.map, 'map', owner);
              push(m.normalMap, 'normalMap', owner);
              push(m.roughnessMap, 'roughnessMap', owner);
              push(m.metalnessMap, 'metalnessMap', owner);
              push(m.aoMap, 'aoMap', owner);
              push(m.emissiveMap, 'emissiveMap', owner);
              push(m.alphaMap, 'alphaMap', owner);
            });
          });
          return texSet;
        }

        mesh.traverse(function(n) {
          if (n.isMesh || n.isSkinnedMesh) {
            meshCount += 1;
            n.castShadow = true;
            n.receiveShadow = true;
            // AO maps need uv2 in Three.js; copy uv -> uv2 when missing.
            try {
              if (n.geometry && n.geometry.attributes && n.geometry.attributes.uv && !n.geometry.attributes.uv2) {
                n.geometry.setAttribute('uv2', n.geometry.attributes.uv);
              }
            } catch (_) {}
            if (n.material) {
              var mats = Array.isArray(n.material) ? n.material : [n.material];
              mats.forEach(function(m) {
                if (!m) return;
                matCount += 1;
                // Texture pipeline fixes: color space + filtering.
                tuneMaterialTextures(m);

                // Material overrides: skin + eyes.
                var nodeName = n.name ? String(n.name) : '';
                var matName = m.name ? String(m.name) : '';
                var isEye = isLikelyEyeName(nodeName) || isLikelyEyeName(matName) || looksEyeByTexture(m);
                var isSkin = isLikelySkinMaterialName(matName) && !isEye;
                if (isSkin) {
                  applySkinTuning(m);
                } else if (isEye) {
                  // Replace material on mesh (clone if needed).
                  var tunedMat = applyEyeTuning(n, m);
                  if (tunedMat !== m) {
                    if (Array.isArray(n.material)) {
                      for (var mi = 0; mi < n.material.length; mi += 1) {
                        if (n.material[mi] === m) n.material[mi] = tunedMat;
                      }
                    } else {
                      n.material = tunedMat;
                    }
                  }
                }
                var op = (typeof m.opacity === 'number') ? m.opacity : 1.0;
                if (m.transparent) transparentCount += 1;
                minOpacity = Math.min(minOpacity, op);
                maxOpacity = Math.max(maxOpacity, op);
              });
            }
            if (n.geometry) {
              if (!n.geometry.boundingBox) n.geometry.computeBoundingBox();
              if (!n.geometry.boundingSphere) n.geometry.computeBoundingSphere();
            }
          }
        });
        console.log('GLB loaded -> meshesFound:', meshCount);
        root.add(mesh);
        var normDebug = normalizeModelOrientation(root) || {};
        var box = new THREE.Box3().setFromObject(root);
        var size = new THREE.Vector3();
        box.getSize(size);
        var maxDim = Math.max(size.x, size.y, size.z);
        if (!isFinite(maxDim) || maxDim < 1e-6) {
          maxDim = 1.0;
        }
        // Keep avatar larger in frame for better readability.
        var distance = maxDim * 1.45;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.position.set(0, maxDim * 0.58, distance);
        camera.near = Math.max(0.01, distance * 0.01);
        camera.far = Math.max(100, distance * 3);
        camera.updateProjectionMatrix();
        camera.lookAt(0, maxDim * 0.54, 0);
        updateRotation();
        // Debug: list textures + resolution + colorspace/encoding.
        var texturesDebug = [];
        var lowResWarnings = [];
        try {
          texturesDebug = buildTextureDebugList(mesh).slice(0, 80);
          for (var ti = 0; ti < texturesDebug.length; ti += 1) {
            var td = texturesDebug[ti];
            if (td && td.width && td.height) {
              // Heuristic: <1K on a face/skin-like texture is likely too soft.
              var owner = (td.owner || '').toLowerCase();
              var usage = (td.usage || '').toLowerCase();
              var looksSkin = owner.indexOf('skin') !== -1 || owner.indexOf('face') !== -1 || owner.indexOf('head') !== -1;
              if (looksSkin && usage === 'map' && (td.width < 1024 || td.height < 1024)) {
                lowResWarnings.push({ owner: td.owner, usage: td.usage, width: td.width, height: td.height });
              }
            }
          }
          if (lowResWarnings.length) {
            console.warn('Low-res skin baseColor textures detected:', lowResWarnings);
          }
          console.log('Texture debug list:', texturesDebug);
        } catch (_) {}
        if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
          post('debug', {
            meshesFound: meshCount,
            bbox: { x: size.x, y: size.y, z: size.z },
            materials: { count: matCount, transparent: transparentCount, minOpacity: minOpacity, maxOpacity: maxOpacity },
            bboxBefore: normDebug.bboxBefore || null,
            bboxAfter: normDebug.bboxAfter || null,
            rootTransformBefore: normDebug.rootTransformBefore || null,
            rootTransformAfter: normDebug.rootTransformAfter || null,
            orientationFixApplied: !!normDebug.orientationFixApplied,
            scaleFixApplied: !!normDebug.scaleFixApplied,
            appliedRotationFix: !!normDebug.appliedRotationFix,
            appliedScaleFix: !!normDebug.appliedScaleFix,
            appliedRotationRadians: Number(normDebug.appliedRotationRadians || 0),
            heightAxisDetected: normDebug.heightAxisDetected || null,
            appliedRotAxis: normDebug.appliedRotAxis || null,
            appliedRotRad: Number(normDebug.appliedRotRad || 0),
            bboxAfterRotation: normDebug.bboxAfterRotation || null,
            detectedUpAxis: normDebug.detectedUpAxis || 'Y',
            appliedScaleFactor: Number(normDebug.appliedScaleFactor || 1),
            groundOffsetApplied: Number(normDebug.groundOffsetApplied || 0),
            orientationNormalized: !!normDebug.orientationNormalized,
            neutralLighting: !!neutralLighting,
            textures: texturesDebug,
            texturesCount: texturesDebug.length,
            lowResWarnings: lowResWarnings.slice(0, 10)
          });
        }
      }, undefined, function(e) {
        console.error('GLB load error', e);
        post('error', {
          stage: 'gltf.load',
          url: glbUrl,
          message: (e && e.message) ? String(e.message) : String(e),
          name: e && e.name ? String(e.name) : null
        });
      });

      clock = new THREE.Clock();
      animate();
    }

    function updateRotation() {
      var root = scene ? scene.getObjectByName('avatarRoot') : null;
      if (!root) return;
      var baseX = (root.userData && typeof root.userData.baseRotX === 'number') ? root.userData.baseRotX : 0.0;
      var baseZ = (root.userData && typeof root.userData.baseRotZ === 'number') ? root.userData.baseRotZ : 0.0;
      root.rotation.set(baseX, (rotationYDeg * Math.PI) / 180, baseZ);
    }

    window.setRotation = function(deg) {
      rotationYDeg = deg;
      updateRotation();
    };

    function notifyRotation() {
      if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
        window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'rotation', deg: rotationYDeg }));
      }
    }

    document.addEventListener('touchstart', function(e) {
      if (e.touches.length !== 1) return;
      touchStartX = e.touches[0].clientX;
      touchStartDeg = rotationYDeg;
    }, { passive: true });
    document.addEventListener('touchmove', function(e) {
      if (e.touches.length !== 1) return;
      var dx = e.touches[0].clientX - touchStartX;
      rotationYDeg = (touchStartDeg + dx * 0.5) % 360;
      if (rotationYDeg < 0) rotationYDeg += 360;
      updateRotation();
      notifyRotation();
    }, { passive: true });

    function animate() {
      requestAnimationFrame(animate);
      if (renderer && scene && camera) renderer.render(scene, camera);
    }

    (async function boot() {
      post('status', { stage: 'boot', message: 'Laster Three.js…' });
      try {
        await loadScript(THREE_CDN);
        post('status', { stage: 'boot', message: 'Laster GLTFLoader…' });
        await loadScript(GLTFLOADER_CDN);
        post('status', { stage: 'boot', message: 'Laster studio environment…' });
        await loadScript(ROOMENV_CDN);
        post('status', { stage: 'boot', message: 'Laster HDR loader…' });
        await loadScript(RGBELOADER_CDN);
        post('status', { stage: 'boot', message: 'Init renderer…' });
        init();
        post('status', { stage: 'boot', message: 'Init OK' });
      } catch (err) {
        console.error('Boot error', err);
        post('error', { stage: 'boot', message: String(err && err.message ? err.message : err) });
      }
    })();
  </script>
</body>
</html>
`;

export default function GLBViewer({ glbUrl, rotationDeg = 0, onRotationChange, style, neutralLighting = true }) {
  const webRef = useRef(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [viewerError, setViewerError] = useState(null);
  const [viewerStatus, setViewerStatus] = useState(null);

  useEffect(() => {
    if (glbUrl) console.log('Rendering avatar from:', glbUrl);
  }, [glbUrl]);

  useEffect(() => {
    if (!webRef.current || !glbUrl) return;
    const deg = ((Number(rotationDeg) % 360) + 360) % 360;
    webRef.current.injectJavaScript(
      `window.setRotation && window.setRotation(${deg}); true;`
    );
  }, [rotationDeg, glbUrl]);

  const handleMessage = (event) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);
      if (data && data.type === 'rotation' && typeof data.deg === 'number' && onRotationChange) {
        const deg = ((data.deg % 360) + 360) % 360;
        onRotationChange(deg);
      } else if (data && data.type === 'debug') {
        setDebugInfo(data);
        setViewerError(null);
        if (__DEV__ && data.textures && Array.isArray(data.textures)) {
          console.log('[GLBViewer] textures (name/usage/res/colorspace):', data.textures);
        }
        if (__DEV__ && data.lowResWarnings && Array.isArray(data.lowResWarnings) && data.lowResWarnings.length) {
          console.warn('[GLBViewer] low-res warnings:', data.lowResWarnings);
        }
      } else if (data && data.type === 'status') {
        setViewerStatus(data);
      } else if (data && data.type === 'net') {
        setViewerStatus(data);
      } else if (data && data.type === 'error') {
        setViewerError(data);
      }
    } catch (_) {}
  };

  if (!glbUrl) return <View style={[styles.container, style]} />;

  const html = makeHTML(glbUrl, !!neutralLighting);

  return (
    <View style={[styles.container, style]}>
      <WebView
        key={`${glbUrl}`}
        ref={webRef}
        source={{ html }}
        style={styles.web}
        scrollEnabled={false}
        javaScriptEnabled={true}
        originWhitelist={['*']}
        mixedContentMode="compatibility"
        onMessage={handleMessage}
        onError={(e) => console.warn('GLBViewer WebView error', e.nativeEvent)}
      />
      {__DEV__ && (viewerError || viewerStatus) ? (
        <View pointerEvents="none" style={styles.debugOverlay}>
          <View style={styles.debugBox}>
            {viewerError ? (
              <Text style={styles.debugText} numberOfLines={4}>
                error: {String(viewerError.stage || '')} {String(viewerError.message || '')}
              </Text>
            ) : null}
            {!viewerError && viewerStatus ? (
              <Text style={styles.debugText} numberOfLines={2}>
                {String(viewerStatus.stage || 'status')}: {String(viewerStatus.message || viewerStatus.status || '')}{viewerStatus.status ? ` (${viewerStatus.status})` : ''}
              </Text>
            ) : null}
          </View>
        </View>
      ) : null}
      {__DEV__ && debugInfo && (
        <View pointerEvents="none" style={styles.debugOverlay}>
          <View style={styles.debugBox}>
            <Text style={styles.debugText}>
              meshes: {debugInfo.meshesFound}
              {debugInfo.bbox ? ` | bbox: ${Number(debugInfo.bbox.x).toFixed(3)}/${Number(debugInfo.bbox.y).toFixed(3)}/${Number(debugInfo.bbox.z).toFixed(3)}` : ''}
              {debugInfo.heightAxisDetected ? ` | hAxis ${String(debugInfo.heightAxisDetected)}` : ''}
              {debugInfo.appliedRotAxis ? ` | rotAxis ${String(debugInfo.appliedRotAxis)}` : ''}
              {typeof debugInfo.appliedRotRad === 'number' ? ` | rotRad ${Number(debugInfo.appliedRotRad).toFixed(3)}` : ''}
              {debugInfo.detectedUpAxis ? ` | up ${String(debugInfo.detectedUpAxis)}` : ''}
              {debugInfo.bboxAfterRotation ? ` | bboxRot: ${Number(debugInfo.bboxAfterRotation.x).toFixed(2)}/${Number(debugInfo.bboxAfterRotation.y).toFixed(2)}/${Number(debugInfo.bboxAfterRotation.z).toFixed(2)}` : ''}
              {debugInfo.bboxBefore && debugInfo.bboxAfter ? ` | norm: ${Number(debugInfo.bboxBefore.x).toFixed(2)}/${Number(debugInfo.bboxBefore.y).toFixed(2)}/${Number(debugInfo.bboxBefore.z).toFixed(2)} -> ${Number(debugInfo.bboxAfter.x).toFixed(2)}/${Number(debugInfo.bboxAfter.y).toFixed(2)}/${Number(debugInfo.bboxAfter.z).toFixed(2)}` : ''}
              {typeof debugInfo.appliedScaleFactor === 'number' ? ` | sF ${Number(debugInfo.appliedScaleFactor).toFixed(4)}` : ''}
              {typeof debugInfo.groundOffsetApplied === 'number' ? ` | gOff ${Number(debugInfo.groundOffsetApplied).toFixed(4)}` : ''}
              {typeof debugInfo.neutralLighting === 'boolean' ? ` | neutralLight ${debugInfo.neutralLighting ? '1' : '0'}` : ''}
              {typeof debugInfo.orientationFixApplied === 'boolean' ? ` | rotFix ${debugInfo.orientationFixApplied ? '1' : '0'}` : ''}
              {typeof debugInfo.scaleFixApplied === 'boolean' ? ` | scaleFix ${debugInfo.scaleFixApplied ? '1' : '0'}` : ''}
              {debugInfo.materials ? ` | mats: ${debugInfo.materials.count} (transp ${debugInfo.materials.transparent}, op ${Number(debugInfo.materials.minOpacity).toFixed(2)}–${Number(debugInfo.materials.maxOpacity).toFixed(2)})` : ''}
              {typeof debugInfo.texturesCount === 'number' ? ` | tex: ${debugInfo.texturesCount}` : ''}
            </Text>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: 360,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    overflow: 'hidden',
  },
  web: {
    flex: 1,
    backgroundColor: 'transparent',
    ...(Platform.OS === 'android' ? { opacity: 0.99 } : {}),
  },
  debugOverlay: {
    position: 'absolute',
    left: 6,
    right: 6,
    bottom: 6,
  },
  debugBox: {
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  debugText: {
    color: '#fff',
    fontSize: 10,
  },
});
